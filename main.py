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

DEFAULT_SYSTEM = (
    "你负责决定：现在要不要主动去找对方说话，以及过多久再找他。只输出一个 JSON，不要解释。"
)

DECISION_TEMPLATE = """当前时间：{now}
今天你已经主动找过他 {used} 次，上限 {cap} 次。
距离他最后一次说话：{idle_desc}
你们最后几句对话：
{recent}

请判断：你想不想主动去找他？如果不想，就表示这次不打扰。
如果想，你打算多久之后去找他（{minm}~{maxm} 分钟之间）？

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


@register("self_schedule", "zhishimaomao", "让她自己决定何时主动找你", "0.3.0")
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
        self._known_umos: set[str] = set(self.config.get("target_umos") or [])

        now = dt.datetime.now(TZ)
        for umo in self._known_umos:
            self._last_msg_at[umo] = now
        # 列表为空时，从最近的会话记录里自动发现私聊，供心跳使用
        if not self._known_umos:
            try:
                self._seed_from_db(now)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[self_schedule] 自动发现会话失败: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._hb_task = loop.create_task(self._heartbeat_loop())
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

        self._known_umos.add(umo)
        self._last_msg_at[umo] = dt.datetime.now(TZ)
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

    # ---------- 心跳 ----------
    async def _heartbeat_loop(self):
        while True:
            try:
                await asyncio.sleep(
                    max(5, int(self.config.get("heartbeat_check_minutes", 60))) * 60
                )
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
    def _seed_from_db(self, now: dt.datetime) -> None:
        """从 AstrBot 的会话表里找出最近的私聊，作为心跳的初始关注对象。"""
        con = sqlite3.connect(f"file:{PERSONA_DB}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "select user_id, updated_at from conversations "
                "where user_id like '%FriendMessage%' "
                "order by updated_at desc limit 10"
            ).fetchall()
        finally:
            con.close()
        for umo, updated in rows:
            if not umo or ":FriendMessage:" not in str(umo):
                continue
            self._known_umos.add(str(umo))
            ts = None
            try:
                ts = dt.datetime.strptime(str(updated)[:19], "%Y-%m-%d %H:%M:%S")
                ts = ts.replace(tzinfo=TZ)
            except Exception:
                ts = None
            self._last_msg_at[str(umo)] = ts or now
        if self._known_umos:
            logger.info(
                f"[self_schedule] 自动发现 {len(self._known_umos)} 个私聊会话用于心跳"
            )

    def _enabled(self) -> bool:
        return bool(self.config.get("enable", True))

    def _umo_allowed(self, umo: str) -> bool:
        targets = self.config.get("target_umos") or []
        return (not targets) or (umo in targets)

    def _buf(self, umo: str) -> deque:
        if umo not in self._buffer:
            self._buffer[umo] = deque(maxlen=20)
        return self._buffer[umo]

    def _recent_text(self, umo: str, n: int = 8) -> str:
        items = list(self._buf(umo))[-n:]
        return "\n".join(f"{who}：{txt[:100]}" for who, txt in items) or "（还没有对话）"

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
            recent=self._recent_text(umo),
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
            recent=self._recent_text(umo),
        )
        system_prompt = self._persona_prompt() or "你是他的恋人，说话短、语气软。"
        text = (await self._llm(system_prompt, prompt)).strip()
        if not text:
            logger.warning("[self_schedule] 生成为空，放弃发送")
            return
        chain = MessageChain([Plain(text)])
        ok = await self.context.send_message(umo, chain)
        self._last_sent_at[umo] = now
        self._buf(umo).append(("你", text))
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
