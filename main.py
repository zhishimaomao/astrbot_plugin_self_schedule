"""让她自己决定何时主动来找你（V0.3：新增心跳，长时间没聊天也会想起你）。

流程：
1. 对话安静 N 秒 → 模型决定「要不要找他、多久之后」→ 要的话插件自己计时，到点生成并发送。
2. 心跳：每隔一段时间检查，若很久没聊天、没有待发计划、不在免打扰内，也会触发一次判断。
"""

import asyncio
import datetime as dt
import json
import re
import sqlite3
from collections import deque
from zoneinfo import ZoneInfo

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, register

TZ = ZoneInfo("Asia/Shanghai")
PERSONA_DB = "/AstrBot/data/data_v4.db"
STATE_FILE = "/AstrBot/data/plugin_data/astrbot_plugin_self_schedule/state.json"

DEFAULT_SYSTEM = (
    "你负责决定：现在要不要主动去找对方说话，以及过多久再找他。只输出一个 JSON，不要解释。"
)

DECISION_TEMPLATE = """当前时间：{now}
今天你已经主动找过他 {used} 次，上限 {cap} 次。
距离他最后一次说话：{idle_desc}
你上次主动之后，他一直没回你的次数：{unanswered} 次
你们最后几句对话：
{recent}

请判断：你想不想主动去找他？如果不想，就表示这次不打扰。
如果想，你打算多久之后去找他（{minm}~{maxm} 分钟之间）？

注意：
- 如果 unanswered 是 1，说明你上次主动找他他没回。这种情况可以再问一次，
  但换个方式：不要重复同样的问候，可以提一件具体的事、或者直接问他是不是在忙。
- 如果 unanswered 达到 2 次以上，就不要再追问了，等他主动。

只输出这样的 JSON：
{{"minutes": 90, "reason": "一句话说明"}}
或者：
{{"minutes": null, "reason": "一句话说明"}}
"""

SPEAK_TEMPLATE = """现在是 {now}。你之前决定在这个时候主动去找他。
当时的想法：{reason}
你们最后几句对话：
{recent}

现在就用你平时的语气，主动发一条消息给他。一到两句就够，别像定时通知，也别解释你为什么来。"""


@register("self_schedule", "zhishimaomao", "让她自己决定何时主动找你", "0.4.1")
class SelfSchedulePlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self._task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._hb_task: asyncio.Task | None = None
        self._last_umo: str | None = None
        self._buffer: dict[str, deque] = {}
        self._last_sent_at: dict[str, dt.datetime] = {}
        self._last_msg_at: dict[str, dt.datetime] = {}
        self._unanswered: dict[str, int] = {}
        self._last_hb_at: dt.datetime | None = None
        self._known_umos: set[str] = set(self.config.get("target_umos") or [])

        now = dt.datetime.now(TZ)
        # 无论是否手动指定会话，都先用数据库里的真实最后消息时间；
        # 查不到才退回"现在"，避免每次重启把沉默计时清零。
        try:
            self._seed_from_db(now)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_schedule] 读取会话时间失败: {e}")
        try:
            self._load_state()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_schedule] 读取状态失败: {e}")
        for umo in self._known_umos:
            self._last_msg_at.setdefault(umo, now)
        for umo, ts in self._last_msg_at.items():
            logger.info(
                "[self_schedule] 心跳关注 %s，最后消息时间 %s"
                % (umo, ts.strftime("%m-%d %H:%M"))
            )

        self._ensure_heartbeat()

    def _ensure_heartbeat(self) -> None:
        if self._hb_task and not self._hb_task.done():
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._hb_task = loop.create_task(self._heartbeat_loop())
                logger.info(
                    "[self_schedule] 心跳已启动：每 %s 分钟检查，沉默 %s 小时后触发"
                    % (
                        self.config.get("heartbeat_check_minutes", 60),
                        self.config.get("heartbeat_after_hours", 6),
                    )
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_schedule] 心跳启动失败: {e}")


    # ---------- 事件 ----------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        if not self._enabled() or event.get_platform_name() != "aiocqhttp":
            return
        umo = event.unified_msg_origin
        if not self._umo_allowed(umo):
            return
        raw = (event.message_str or "").strip()
        if not raw:
            # 命令/空事件不算聊天
            return
        plain_cmd = raw.lstrip("/").strip()
        if raw.startswith("/") or plain_cmd in ("插件状态", "主动状态"):
            # 管理命令不算聊天，不重置沉默计时
            return

        self._known_umos.add(umo)
        self._last_msg_at[umo] = dt.datetime.now(TZ)
        last_sent = self._last_sent_at.get(umo)
        if last_sent and self._last_msg_at[umo] > last_sent:
            self._unanswered[umo] = 0
        self._save_state()
        self._buf(umo).append(("他", (event.message_str or "").strip()))
        self._last_umo = umo
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._decide_later(umo))

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp):
        try:
            umo = event.unified_msg_origin
            text = (getattr(resp, "completion_text", "") or "").strip()
            if text and umo in self._buffer:
                self._buffer[umo].append(("你", text))
        except Exception:
            pass


    @filter.command("插件状态")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看自主找他插件的运行状态。"""
        umo = event.unified_msg_origin
        now = dt.datetime.now(TZ)
        last = self._last_msg_at.get(umo)
        if last:
            mins = int((now - last).total_seconds() // 60)
            last_desc = f"{last.strftime('%m-%d %H:%M')}（{mins} 分钟前）"
        else:
            last_desc = "未知"

        hb_min = int(self.config.get("heartbeat_check_minutes", 60))
        hb_hours = int(self.config.get("heartbeat_after_hours", 6))
        pending = bool(self._send_task and not self._send_task.done())
        hb_alive = bool(self._hb_task and not self._hb_task.done())
        next_hb = (
            (self._last_hb_at or now) + dt.timedelta(minutes=hb_min)
            if hb_alive
            else None
        )

        lines = [
            "【自主找他 · 状态】",
            f"插件：{'运行中' if self._enabled() else '已停用'}",
            f"关注会话：{umo}",
            f"最后消息：{last_desc}",
            f"沉默阈值：{hb_hours} 小时",
            f"今天已主动：{self._count_today(umo)} / {self.config.get('max_per_day', 3)} 次",
            f"连续未回复：{int(self._unanswered.get(umo, 0))} 次",
            f"待发送计划：{'有' if pending else '无'}",
            f"上下文：会话历史 {len(await self._history_items(umo, 10))} 条",
            f"心跳：{'每 %s 分钟' % hb_min if hb_alive else '未运行'}"
            + (f"，下次约 {next_hb.strftime('%H:%M')}" if next_hb else ""),
            f"免打扰：{self.config.get('quiet_start', 0)}:00 - {self.config.get('quiet_end', 8)}:00",
            f"她可选时间：{self.config.get('min_minutes', 45)} ~ {self.config.get('max_minutes', 360)} 分钟",
        ]
        yield event.plain_result("\n".join(lines))

    # ---------- 心跳 ----------
    async def _heartbeat_loop(self):
        while True:
            try:
                await asyncio.sleep(
                    max(5, int(self.config.get("heartbeat_check_minutes", 60))) * 60
                )
                self._last_hb_at = dt.datetime.now(TZ)
                await self._heartbeat_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[self_schedule] 心跳异常: {e}")

    async def _heartbeat_tick(self):
        if not self._enabled():
            return
        hours = int(self.config.get("heartbeat_after_hours", 6))
        if hours <= 0:
            return
        now = dt.datetime.now(TZ)
        if self._in_quiet(now):
            return
        for umo in list(self._known_umos):
            if not self._umo_allowed(umo):
                continue
            if self._send_task and not self._send_task.done():
                continue
            last = self._last_msg_at.get(umo)
            if last is None:
                self._last_msg_at[umo] = now
                continue
            if (now - last) < dt.timedelta(hours=hours):
                continue
            logger.info(f"[self_schedule] 心跳触发：已 {hours} 小时没聊天，让她判断一次")
            await self._run_decision(umo)
            return

    # ---------- 基础 ----------
    def _load_state(self) -> None:
        import os

        if not os.path.exists(STATE_FILE):
            return
        data = json.load(open(STATE_FILE, encoding="utf-8"))
        for umo, item in (data or {}).items():
            try:
                ts = item.get("last_msg")
                if ts:
                    self._last_msg_at[umo] = dt.datetime.fromisoformat(ts)
                self._unanswered[umo] = int(item.get("unanswered", 0))
                sent_ts = item.get("last_sent")
                if sent_ts:
                    self._last_sent_at[umo] = dt.datetime.fromisoformat(sent_ts)
                self._known_umos.add(umo)
            except Exception:
                continue
        if self._known_umos:
            logger.info(f"[self_schedule] 从状态文件恢复 {len(self._known_umos)} 个会话")

    def _save_state(self) -> None:
        import os

        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            data = {}
            for umo in self._known_umos:
                ts = self._last_msg_at.get(umo)
                sent = self._last_sent_at.get(umo)
                data[umo] = {
                    "last_msg": ts.isoformat() if ts else None,
                    "last_sent": sent.isoformat() if sent else None,
                    "unanswered": int(self._unanswered.get(umo, 0)),
                }
            json.dump(data, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass

    def _seed_from_db(self, now: dt.datetime) -> None:
        """从 AstrBot 的会话表里读取最近私聊，以及它们真实的最后消息时间。"""
        con = sqlite3.connect(f"file:{PERSONA_DB}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "select user_id, updated_at from conversations "
                "where user_id like '%FriendMessage%' "
                "order by updated_at desc limit 10"
            ).fetchall()
        finally:
            con.close()

        targets = set(self.config.get("target_umos") or [])
        for umo, updated in rows:
            umo = str(umo or "")
            if ":FriendMessage:" not in umo:
                continue
            if targets and umo not in targets:
                continue
            self._known_umos.add(umo)
            ts = None
            try:
                ts = dt.datetime.strptime(str(updated)[:19], "%Y-%m-%d %H:%M:%S")
                ts = ts.replace(tzinfo=TZ)
            except Exception:
                ts = None
            self._last_msg_at[umo] = ts or now

    def _enabled(self) -> bool:
        return bool(self.config.get("enable", True))

    def _umo_allowed(self, umo: str) -> bool:
        targets = self.config.get("target_umos") or []
        return (not targets) or (umo in targets)

    def _buf(self, umo: str) -> deque:
        if umo not in self._buffer:
            self._buffer[umo] = deque(maxlen=20)
        return self._buffer[umo]

    async def _history_items(self, umo: str, n: int = 10) -> list[dict]:
        try:
            cm = self.context.conversation_manager
            cid = await cm.get_curr_conversation_id(umo)
            if not cid:
                return []
            conv = await cm.get_conversation(umo, cid)
            if not conv:
                return []
            raw = conv.history
            data = json.loads(raw) if isinstance(raw, str) else (raw or [])
            return list(data)[-n:]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_schedule] 读取会话历史失败: {e}")
            return []

    async def _recent_text(self, umo: str, n: int = 8) -> str:
        items = await self._history_items(umo, n)
        lines = []
        for m in items:
            role = "他" if m.get("role") == "user" else "你"
            parts = m.get("content")
            if isinstance(parts, list):
                txt = "".join(
                    c.get("text", "") for c in parts
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            else:
                txt = str(parts or "")
            txt = txt.strip()
            if txt:
                lines.append(f"{role}：{txt[:120]}")
        if lines:
            return "\n".join(lines)
        buffered = list(self._buf(umo))[-n:]
        return "\n".join(f"{who}：{txt[:100]}" for who, txt in buffered) or "（还没有对话）"

    async def _append_history(self, umo: str, text: str) -> None:
        try:
            cm = self.context.conversation_manager
            cid = await cm.get_curr_conversation_id(umo)
            if not cid:
                return
            conv = await cm.get_conversation(umo, cid)
            if not conv:
                return
            raw = conv.history
            data = json.loads(raw) if isinstance(raw, str) else list(raw or [])
            data.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
            await cm.update_conversation(umo, cid, history=data)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_schedule] 写回会话历史失败: {e}")

    def _in_quiet(self, when: dt.datetime) -> bool:
        qs = int(self.config.get("quiet_start", 0))
        qe = int(self.config.get("quiet_end", 8))
        if qs == qe:
            return False
        h = when.hour
        return (qs <= h < qe) if qs <= qe else (h >= qs or h < qe)

    # ---------- 决策 ----------
    async def _decide_later(self, umo: str):
        try:
            await asyncio.sleep(int(self.config.get("idle_seconds", 90)))
        except asyncio.CancelledError:
            return
        if umo != self._last_umo:
            return
        try:
            await self._run_decision(umo)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_schedule] 决策失败: {e}")

    async def _run_decision(self, umo: str):
        if self._send_task and not self._send_task.done():
            logger.info("[self_schedule] 已有待发送的计划，跳过")
            return

        if int(self._unanswered.get(umo, 0)) >= 2:
            logger.info("[self_schedule] 连续两次没被回复，先不再打扰")
            return

        used = self._count_today(umo)
        cap = int(self.config.get("max_per_day", 3))
        if used >= cap:
            logger.info(f"[self_schedule] 今天已达上限 {used}/{cap}，跳过")
            return

        now = dt.datetime.now(TZ)
        minm = int(self.config.get("min_minutes", 45))
        maxm = max(minm + 5, int(self.config.get("max_minutes", 360)))
        last = self._last_msg_at.get(umo)
        idle_desc = (
            "%.1f 小时前" % ((now - last).total_seconds() / 3600) if last else "不确定"
        )
        prompt = DECISION_TEMPLATE.format(
            now=now.strftime("%Y-%m-%d %H:%M (%A)"),
            used=used,
            cap=cap,
            idle_desc=idle_desc,
            unanswered=int(self._unanswered.get(umo, 0)),
            recent=await self._recent_text(umo),
            minm=minm,
            maxm=maxm,
        )
        text = await self._llm(
            self.config.get("decision_system_prompt") or DEFAULT_SYSTEM, prompt
        )
        logger.info(f"[self_schedule] 决策输出: {text[:160]}")

        minutes = self._parse_minutes(text)
        if minutes is None:
            logger.info("[self_schedule] 这次决定不找他")
            return

        minutes = max(minm, min(maxm, minutes))
        run_at = self._avoid_quiet(now + dt.timedelta(minutes=minutes))
        reason = self._parse_reason(text)
        wait = max(1, int((run_at - now).total_seconds()))
        logger.info(
            f"[self_schedule] 计划 {run_at.strftime('%m-%d %H:%M')} 主动找他（{wait}s 后）"
        )
        self._send_task = asyncio.create_task(self._later_send(umo, wait, reason))

    async def _later_send(self, umo: str, wait: int, reason: str):
        try:
            await asyncio.sleep(wait)
            await self._speak(umo, reason)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"[self_schedule] 发送失败: {e}")

    async def _speak(self, umo: str, reason: str):
        now = dt.datetime.now(TZ)
        prompt = SPEAK_TEMPLATE.format(
            now=now.strftime("%Y-%m-%d %H:%M (%A)"),
            reason=reason,
            recent=await self._recent_text(umo),
        )
        system_prompt = self._persona_prompt() or "你是他的恋人，说话短、语气软。"
        text = (await self._llm(system_prompt, prompt)).strip()
        if not text:
            logger.warning("[self_schedule] 生成为空，放弃发送")
            return
        chain = MessageChain([Plain(text)])
        ok = await self.context.send_message(umo, chain)
        self._last_sent_at[umo] = now
        self._unanswered[umo] = int(self._unanswered.get(umo, 0)) + 1
        self._buf(umo).append(("你", text))
        await self._append_history(umo, text)
        self._save_state()
        logger.info(f"[self_schedule] 已主动发送（成功={ok}）: {text[:80]}")

    # ---------- 工具 ----------
    async def _llm(self, system_prompt: str, prompt: str) -> str:
        provider_id = str(self.config.get("provider_id") or "").strip()
        if not provider_id:
            provider_id = await self.context.get_current_chat_provider_id(self._last_umo)
        resp = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return (resp.completion_text or "").strip()

    def _persona_prompt(self) -> str:
        try:
            con = sqlite3.connect(f"file:{PERSONA_DB}?mode=ro", uri=True)
            row = con.execute(
                "select system_prompt from personas where persona_id='小鲸'"
            ).fetchone()
            con.close()
            return row[0] if row and row[0] else ""
        except Exception:
            return ""

    def _parse_minutes(self, text: str) -> int | None:
        m = re.search(r"\{.*\}", text, re.S)
        raw = m.group(0) if m else text
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, dict):
            val = data.get("minutes")
            if val is None:
                return None
            try:
                return int(val)
            except Exception:
                return None
        m2 = re.search(r'"minutes"\s*:\s*(\d+)', text)
        return int(m2.group(1)) if m2 else None

    def _parse_reason(self, text: str) -> str:
        m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        return m.group(1) if m else ""

    def _avoid_quiet(self, when: dt.datetime) -> dt.datetime:
        if not self._in_quiet(when):
            return when
        qe = int(self.config.get("quiet_end", 8))
        target = when.replace(hour=qe % 24, minute=10, second=0, microsecond=0)
        if target <= when:
            target += dt.timedelta(days=1)
        return target

    def _count_today(self, umo: str) -> int:
        today = dt.datetime.now(TZ).date()
        return sum(
            1
            for key, when in self._last_sent_at.items()
            if key == umo and when.date() == today
        )
