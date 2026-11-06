"""
i18n.py — the interface language of NexusAsk (English / 中文).

    import i18n
    from i18n import t

    i18n.init(st.session_state.get("user_email"))   # once per run, at the top of app.py
    st.button(t("nav.new_chat"))
    st.caption(t("chat.loaded", files=names, n=count))

* The language comes from the user's Settings > Language (user_settings/<email>.json).
  "auto" follows the browser's language; before login it is always auto.
* Every key MUST exist in BOTH languages (test_i18n.py checks this, including {placeholders}).
* Not translated on purpose: the brand name NexusAsk, log lines, developer-only info,
  prompts sent to the model.
"""
import json
import os
import re

import streamlit as st

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_settings")

STRINGS = {
    "en": {
        # ---- sign in ----
        "auth.login_tab": "Login", "auth.register_tab": "Register",
        "auth.login_title": "Login", "auth.create_title": "Create an account",
        "auth.email": "Email", "auth.password": "Password", "auth.name": "Your name", "auth.confirm": "Confirm password",
        "auth.enter_both": "Enter your email and password.",
        "auth.invalid_login": "Invalid email or password.",
        "auth.all_required": "All fields are required.",
        "auth.invalid_email": "Invalid email.",
        "auth.mismatch": "Passwords don't match.",
        "auth.too_short": "Password is too short.",
        "auth.exists": "An account with this email already exists.",
        "auth.created": "Account created! Switch to Login.",
        # ---- sidebar ----
        "nav.new_chat": "New chat", "nav.library": "Library", "nav.chats": "Chats",
        "nav.no_chats": "No conversations yet", "nav.search_help": "Search chats",
        "nav.saving": "Saving conversation…",
        "nav.delete": "Delete", "nav.rename": "Rename", "nav.save": "Save", "nav.cancel": "Cancel",
        # ---- account menu ----
        "acct.settings": "Settings", "acct.language": "Language", "acct.help": "Get help", "acct.logout": "Log out",
        # ---- search dialog ----
        "search.title": "Search chats", "search.placeholder": "Search chats (press Enter)",
        "search.results": "Results", "search.none": "No matching chats",
        "search.recent_searches": "Recent searches", "search.recent_chats": "Recent chats",
        "search.clear_all": "Clear all", "search.remove": "Remove from history",
        # ---- settings dialog ----
        "set.title": "Settings", "set.account": "Account", "set.display": "Display",
        "set.dev_toggle": "Show developer info under answers",
        "set.dev_help": "Token count, cache status and the LangSmith trace.",
        "set.data": "Your data",
        "set.memory_note": "NexusAsk remembers facts from past chats to personalize answers.",
        "set.clear_memory": "Clear long-term memory", "set.clear_chats": "Clear chat history",
        "set.clear_searches": "Clear search history",
        "set.cannot_undo": "This cannot be undone.", "set.yes": "Yes",
        "set.memory_cleared": "Long-term memory cleared", "set.chats_cleared": "Chat history cleared",
        "set.searches_cleared": "Search history cleared",
        # ---- language dialog ----
        "lang.title": "Language",
        "lang.note": "Language of the interface and of NexusAsk's answers. Auto follows your browser for the interface and matches the language of your question for answers.",
        "lang.auto": "Auto", "lang.en": "English", "lang.zh": "中文（简体）",
        # ---- get help ----
        "help.title": "Get help",
        "help.body": (
            "**Welcome to NexusAsk**\n\n"
            "An assistant that answers from your documents, the web and Wikipedia, and shows you what it did along the way.\n\n"
            "**What you can do**\n\n"
            "- **Ask anything.** NexusAsk picks the right tool by itself: your documents, web search, Wikipedia or a calculator.\n"
            "- **Chat about your files.** Use the + next to the message box, or drag and drop. Supported: PDF, Word, PowerPoint, Excel, TXT, Markdown, CSV.\n"
            "- **Find files again** in Library. Everything you upload is kept there.\n"
            "- **Come back to a chat.** Chats are saved automatically. Use New chat to start fresh, the three dots on a chat to rename or delete it, and the magnifier to search every chat.\n"
            "- **See how an answer was made.** Open Show steps under a reply to see each step and how long it took.\n"
            "- **Copy or retry** an answer with the small icons under it.\n"
            "- **Get personalized answers.** NexusAsk remembers useful facts from your past chats. You can clear this in Settings.\n\n"
            "**Good to know**\n\n"
            "- Mention a website or say \"search the web\" to make it look online first.\n"
            "- Answers that use the web or Wikipedia end with a reminder that they are AI-generated, so check important facts.\n"
            "- Very similar questions at the start of a chat may be answered from a saved earlier answer, which is faster.\n"
            "- Requests that contain certain sensitive words are declined."
        ),
        # ---- chat ----
        "chat.greeting": "Hi {name}, what's on your mind?",
        "chat.ask_anything": "Ask anything",
        "chat.ph_docs": "Ask a question about the documents…",
        "chat.ph_any": "Ask anything (Wikipedia + web search)…",
        "chat.followup": "Ask a follow-up…", "chat.continue_asking": "Continue asking:",
        "chat.loaded": "Loaded: {files} · {n} chunks",
        "chat.uploaded": "File uploaded! {n} file(s) loaded.",
        "chat.reading_files": "Reading files…",
        "chat.loading_model": "Loading embedding model (first run downloads ~470 MB)…",
        "upload.caption": "Drag & drop or click to upload (pdf, docx, pptx, xlsx, txt, md, csv)",
        "upload.help": "Upload files",
        "footer.disclaimer": "NexusAsk can make mistakes. Please double check.",
        # ---- library / preview / files ----
        "lib.empty": "No files in your library yet. Upload a file to get started.",
        "lib.back_chat": "Back to chat", "lib.back_lib": "Back to library",
        "lib.not_found": "File data not found for {name}.",
        "lib.no_preview": "Preview not available for {suffix}.",
        "lib.delete_title": "Delete file", "lib.delete_confirm": "Are you sure you want to delete {name}?",
        "file.unsupported": "Unsupported file type: {suffix}",
        "file.no_text": "No text found in {name}.",
        "file.read_failed": "Couldn't read {name}: {error}",
        "review.back": "Back to current", "review.continue": "Continue",
        # ---- confirm dialogs ----
        "dlg.clear_all_title": "Clear all history",
        "dlg.clear_all_body": "This will permanently delete ALL your saved conversations. This cannot be undone.",
        "dlg.delete_all": "Delete all",
        "dlg.logout_title": "Log out",
        "dlg.logout_body": "Your current conversation will be saved and your long-term memory updated. Log out now?",
        "dlg.delete_conv_title": "Delete conversation", "dlg.delete_conv_body": "Delete this conversation?",
        "dlg.clear_memory_title": "Clear memory",
        "dlg.clear_memory_body": "Clear all long-term memories NexusAsk has learned about you?",
        "dlg.clear_memory_btn": "Clear memory",
        # ---- one assistant turn ----
        "turn.thinking": "Nexus is thinking…", "turn.done": "Done", "turn.failed": "Failed",
        "turn.error": "Nexus couldn't reach the model. Please try again.",
        "turn.show_steps": "Show steps",
        "turn.interrupted": "Response interrupted.", "turn.continue": "Click to continue",
        "action.retry": "Retry", "action.copy": "Copy", "action.stop": "Stop",
        # ---- progress labels ----
        "status.rewriting": "Rewriting your question",
        "status.thinking_step": "Thinking (step {step})",
        "status.retrying": "Model is slow, retrying ({n}/{total})",
        "status.ready": "Answer ready", "status.cached": "Answer from a similar earlier question",
        "status.blocked": "Blocked (sensitive word)",
        "tool.search_docs": "Searching your documents", "tool.search_web": "Searching the web",
        "tool.calculator": "Calculating", "tool.wikipedia_search": "Searching Wikipedia",
        "tool.unknown": "Using {name}",
        "status.tool_done": "{tool} · {n} result(s)",
    },
    "zh": {
        # ---- 登录 ----
        "auth.login_tab": "登录", "auth.register_tab": "注册",
        "auth.login_title": "登录", "auth.create_title": "创建账号",
        "auth.email": "邮箱", "auth.password": "密码", "auth.name": "你的名字", "auth.confirm": "确认密码",
        "auth.enter_both": "请输入邮箱和密码。",
        "auth.invalid_login": "邮箱或密码不正确。",
        "auth.all_required": "请填写所有项目。",
        "auth.invalid_email": "邮箱格式不正确。",
        "auth.mismatch": "两次输入的密码不一致。",
        "auth.too_short": "密码太短。",
        "auth.exists": "该邮箱已注册。",
        "auth.created": "账号已创建！请切换到“登录”。",
        # ---- 侧边栏 ----
        "nav.new_chat": "新对话", "nav.library": "资料库", "nav.chats": "对话",
        "nav.no_chats": "还没有对话", "nav.search_help": "搜索对话",
        "nav.saving": "正在保存对话…",
        "nav.delete": "删除", "nav.rename": "重命名", "nav.save": "保存", "nav.cancel": "取消",
        # ---- 账号菜单 ----
        "acct.settings": "设置", "acct.language": "语言", "acct.help": "获取帮助", "acct.logout": "退出登录",
        # ---- 搜索弹窗 ----
        "search.title": "搜索对话", "search.placeholder": "搜索对话（按回车）",
        "search.results": "搜索结果", "search.none": "没有匹配的对话",
        "search.recent_searches": "最近搜索", "search.recent_chats": "最近对话",
        "search.clear_all": "全部清除", "search.remove": "从历史中移除",
        # ---- 设置弹窗 ----
        "set.title": "设置", "set.account": "账号", "set.display": "显示",
        "set.dev_toggle": "在回答下方显示开发者信息",
        "set.dev_help": "token 数量、缓存状态和 LangSmith 追踪。",
        "set.data": "你的数据",
        "set.memory_note": "NexusAsk 会记住过去对话中的信息，让回答更贴合你。",
        "set.clear_memory": "清除长期记忆", "set.clear_chats": "清除聊天记录",
        "set.clear_searches": "清除搜索历史",
        "set.cannot_undo": "此操作无法撤销。", "set.yes": "确定",
        "set.memory_cleared": "长期记忆已清除", "set.chats_cleared": "聊天记录已清除",
        "set.searches_cleared": "搜索历史已清除",
        # ---- 语言弹窗 ----
        "lang.title": "语言",
        "lang.note": "界面和 NexusAsk 回答所使用的语言。选择“自动”时，界面跟随浏览器语言，回答则与你提问的语言一致。",
        "lang.auto": "自动", "lang.en": "English", "lang.zh": "中文（简体）",
        # ---- 获取帮助 ----
        "help.title": "获取帮助",
        "help.body": (
            "**欢迎使用 NexusAsk**\n\n"
            "一个能根据你的文档、网络和维基百科来回答问题的助手，并且会告诉你它每一步做了什么。\n\n"
            "**你可以做什么**\n\n"
            "- **随便提问。** NexusAsk 会自己选择合适的工具：你的文档、联网搜索、维基百科或计算器。\n"
            "- **和文件对话。** 点消息框旁边的 +，或直接拖放文件。支持 PDF、Word、PowerPoint、Excel、TXT、Markdown、CSV。\n"
            "- **在资料库里找回文件。** 你上传的所有文件都保存在那里。\n"
            "- **回到之前的对话。** 对话会自动保存。点“新对话”重新开始，点对话右边的三个点可以重命名或删除，点放大镜可以搜索所有对话。\n"
            "- **查看回答是怎么得出的。** 点回答下方的“查看步骤”，可以看到每一步以及耗时。\n"
            "- **复制或重试**回答：用回答下方的小图标。\n"
            "- **获得更贴合你的回答。** NexusAsk 会记住你过去对话中有用的信息，可以在“设置”里清除。\n\n"
            "**小提示**\n\n"
            "- 提到某个网站，或说“联网搜索”，它会先去网上查。\n"
            "- 用到网络或维基百科的回答末尾会提醒“由AI生成”，重要信息请自行核实。\n"
            "- 对话开头的相似问题，可能直接用之前保存的回答，速度更快。\n"
            "- 含有某些敏感词的请求会被拒绝。"
        ),
        # ---- 聊天 ----
        "chat.greeting": "你好，{name}，有什么想聊的？",
        "chat.ask_anything": "问点什么吧",
        "chat.ph_docs": "就文档提问…",
        "chat.ph_any": "随便问（维基百科 + 联网搜索）…",
        "chat.followup": "继续提问…", "chat.continue_asking": "继续提问：",
        "chat.loaded": "已加载：{files} · {n} 个片段",
        "chat.uploaded": "文件已上传！已加载 {n} 个文件。",
        "chat.reading_files": "正在读取文件…",
        "chat.loading_model": "正在加载嵌入模型（首次运行会下载约 470 MB）…",
        "upload.caption": "拖放或点击上传（pdf、docx、pptx、xlsx、txt、md、csv）",
        "upload.help": "上传文件",
        "footer.disclaimer": "NexusAsk 可能会出错，请注意核实。",
        # ---- 资料库 / 预览 / 文件 ----
        "lib.empty": "资料库里还没有文件。上传一个文件开始吧。",
        "lib.back_chat": "返回对话", "lib.back_lib": "返回资料库",
        "lib.not_found": "找不到文件 {name} 的数据。",
        "lib.no_preview": "暂不支持预览 {suffix} 文件。",
        "lib.delete_title": "删除文件", "lib.delete_confirm": "确定要删除 {name} 吗？",
        "file.unsupported": "不支持的文件类型：{suffix}",
        "file.no_text": "在 {name} 中没有找到文字。",
        "file.read_failed": "无法读取 {name}：{error}",
        "review.back": "返回当前对话", "review.continue": "继续这个对话",
        # ---- 确认弹窗 ----
        "dlg.clear_all_title": "清除全部历史",
        "dlg.clear_all_body": "这会永久删除你保存的全部对话，无法撤销。",
        "dlg.delete_all": "全部删除",
        "dlg.logout_title": "退出登录",
        "dlg.logout_body": "当前对话会被保存，长期记忆也会更新。现在退出登录吗？",
        "dlg.delete_conv_title": "删除对话", "dlg.delete_conv_body": "删除这个对话吗？",
        "dlg.clear_memory_title": "清除记忆",
        "dlg.clear_memory_body": "清除 NexusAsk 记住的所有关于你的长期记忆吗？",
        "dlg.clear_memory_btn": "清除记忆",
        # ---- 一轮回答 ----
        "turn.thinking": "Nexus 正在思考…", "turn.done": "完成", "turn.failed": "失败",
        "turn.error": "Nexus 暂时连不上模型，请重试。",
        "turn.show_steps": "查看步骤",
        "turn.interrupted": "回答已中断。", "turn.continue": "点击继续",
        "action.retry": "重试", "action.copy": "复制", "action.stop": "停止",
        # ---- 进度文字 ----
        "status.rewriting": "正在改写问题",
        "status.thinking_step": "思考中（第 {step} 步）",
        "status.retrying": "模型响应较慢，正在重试（{n}/{total}）",
        "status.ready": "回答已就绪", "status.cached": "来自相似的历史提问",
        "status.blocked": "已拦截（含敏感词）",
        "tool.search_docs": "正在搜索你的文档", "tool.search_web": "正在联网搜索",
        "tool.calculator": "正在计算", "tool.wikipedia_search": "正在搜索维基百科",
        "tool.unknown": "正在使用 {name}",
        "status.tool_done": "{tool} · {n} 条结果",
    },
}
LANG_KEY = "_ui_lang"


def _browser_lang():
    """'zh' when the browser's language is Chinese, otherwise 'en'."""
    try:
        loc = st.context.locale
    except Exception:
        loc = None
    return "zh" if loc and str(loc).lower().startswith("zh") else "en"


def resolve(setting):
    """Settings value ("auto" / "English" / "Simplified Chinese") -> "en" or "zh"."""
    if setting == "English":
        return "en"
    if setting == "Simplified Chinese":
        return "zh"
    return _browser_lang()


def _read_setting(email):
    if not email:
        return "auto"
    path = os.path.join(SETTINGS_DIR, f"{re.sub(r'[^a-zA-Z0-9._-]', '_', email)}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("language", "auto") if isinstance(data, dict) else "auto"
    except (OSError, ValueError):
        return "auto"


def init(email=None):
    """Call once per run, at the top of app.py (email is None before login)."""
    st.session_state[LANG_KEY] = resolve(_read_setting(email))
    return st.session_state[LANG_KEY]


def lang():
    return st.session_state.get(LANG_KEY, "en")


def t(key, **kw):
    """Text for `key` in the current language (falls back to English, then to the key itself)."""
    s = STRINGS.get(lang(), STRINGS["en"]).get(key) or STRINGS["en"].get(key) or key
    return s.format(**kw) if kw else s


def tool_label(name):
    """Friendly, translated name of an agent tool (search_docs, search_web, ...)."""
    key = f"tool.{name}"
    return t(key) if key in STRINGS["en"] else t("tool.unknown", name=name)


def css_vars():
    """<style> block for the texts that live in style.css (the disclaimer under the chat box)."""
    text = t("footer.disclaimer").replace("\\", "\\\\").replace('"', '\\"')
    return f'<style>:root {{ --disclaimer: "{text}"; }}</style>'




