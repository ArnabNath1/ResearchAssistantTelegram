"""
Telegram Bot Handlers — All command and message handlers.
Provides a rich, interactive research assistant experience.
"""

import io
import asyncio
from loguru import logger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
)
from telegram.constants import ParseMode, ChatAction
from research_engine import research_engine

# Conversation states
WAITING_FOR_PDF = 1
WAITING_FOR_ARXIV_ID = 2
WAITING_FOR_QUESTION = 3
WAITING_FOR_EQUATION = 4
WAITING_FOR_TOPIC = 5

# Per-user state store (in-memory; use Redis for production scale)
USER_STATE: dict[int, dict] = {}


def get_user_state(user_id: int) -> dict:
    if user_id not in USER_STATE:
        USER_STATE[user_id] = {"chat_history": [], "last_paper_id": None}
    return USER_STATE[user_id]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )


async def send_long_message(update: Update, text: str, parse_mode: str = ParseMode.MARKDOWN):
    """
    Send long messages by splitting into chunks ≤ 4096 chars.
    Telegram's limit is 4096 characters per message.
    """
    chunk_size = 4000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            # Fallback without markdown if parsing fails
            await update.message.reply_text(chunk)
        if len(chunks) > 1:
            await asyncio.sleep(0.3)  # Small delay between chunks


def main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main action keyboard."""
    buttons = [
        [
            InlineKeyboardButton("📄 Analyze PDF", callback_data="menu_analyze"),
            InlineKeyboardButton("🔍 Search Papers", callback_data="menu_search"),
        ],
        [
            InlineKeyboardButton("🕳️ Find Gaps", callback_data="menu_gaps"),
            InlineKeyboardButton("🧪 Experiments", callback_data="menu_experiments"),
        ],
        [
            InlineKeyboardButton("💡 Novel Ideas", callback_data="menu_novelty"),
            InlineKeyboardButton("📚 Lit Review", callback_data="menu_litreview"),
        ],
        [
            InlineKeyboardButton("📐 Explain Math", callback_data="menu_equations"),
            InlineKeyboardButton("📦 Datasets", callback_data="menu_datasets"),
        ],
        [
            InlineKeyboardButton("❓ Research Q&A", callback_data="menu_qa"),
            InlineKeyboardButton("🆔 arXiv Fetch", callback_data="menu_arxiv"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    welcome = (
        f"🔬 *Welcome, {user.first_name}! I am your AI Research Assistant.* 🚀\n\n"
        "I am a scientific reasoning engine designed to accelerate your research workflow. "
        "Send me any research PDF or an arXiv ID to get started.\n\n"
        "**🔹 What I can do for you:**\n"
        "• 📄 *Analyze Papers*: Full extraction of methodology, math, and risks.\n"
        "• 🕳️ *Find Gaps*: Detect missing pieces in current research.\n"
        "• 🧪 *Experiments*: Suggest concrete experimental designs.\n"
        "• 💡 *Generate Novelty*: Brainstorm original research directions.\n"
        "• 📚 *Literature Reviews*: Synthesize field-level knowledge.\n"
        "• 📐 *Explain Math*: Break down complex equations.\n\n"
        "**⌨️ Quick Commands:**\n"
        "/start — Open the research menu.\n"
        "/arxiv <id> — Download & analyze an arXiv paper.\n"
        "/search <topic> — Find the latest papers on a subject.\n"
        "/qa <question> — Ask technical questions.\n\n"
        "**💡 Pro Tip:** You can drag and drop any PDF file here to analyze it instantly!"
    )
    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "📖 *Commands*\n\n"
        "`/start` — Welcome screen & menu\n"
        "`/analyze` — Analyze a PDF paper\n"
        "`/arxiv <id>` — Analyze by arXiv ID (e.g., `/arxiv 2303.08774`)\n"
        "`/search <query>` — Search research papers\n"
        "`/gaps` — Find research gaps in uploaded paper\n"
        "`/experiments` — Suggest experiments\n"
        "`/novelty` — Generate novel research ideas\n"
        "`/litreview <topic>` — Generate literature review\n"
        "`/equation <eq>` — Explain a mathematical equation\n"
        "`/datasets <task>` — Recommend datasets\n"
        "`/qa <question>` — Research Q&A (RAG-powered)\n"
        "`/summary` — Quick TL;DR of uploaded paper\n"
        "`/clear` — Clear conversation history\n"
        "`/menu` — Show main menu\n\n"
        "💡 *Tip*: Send a PDF file directly to analyze it instantly!"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the main menu."""
    await update.message.reply_text(
        "🔬 *Research Assistant Menu* — Select an action:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_keyboard(),
    )


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear user conversation history."""
    user_id = update.effective_user.id
    USER_STATE[user_id] = {"chat_history": [], "last_paper_id": None}
    await update.message.reply_text("🗑️ Conversation history cleared!")


# ─────────────────────────────────────────────────────────────────────────────
# PDF HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def pdf_document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle PDF document uploads."""
    doc = update.message.document
    if not doc or doc.mime_type != "application/pdf":
        await update.message.reply_text("❌ Please send a valid PDF file.")
        return

    # Store file_id in user_data to avoid callback_data size limits (64 bytes)
    context.user_data["last_pdf_file_id"] = doc.file_id

    await update.message.reply_text(
        "📥 PDF received! Choose analysis type:",
        reply_markup=InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔬 Full Analysis", callback_data="pdf_analyze"),
                InlineKeyboardButton("📄 Quick Summary", callback_data="pdf_summary"),
            ],
            [
                InlineKeyboardButton("🕳️ Find Gaps", callback_data="pdf_gaps"),
                InlineKeyboardButton("🧪 Experiments", callback_data="pdf_experiments"),
            ],
            [
                InlineKeyboardButton("💡 Novel Ideas", callback_data="pdf_novelty"),
            ],
        ]),
    )


async def _download_pdf(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    """Download a file from Telegram servers."""
    file = await context.bot.get_file(file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def arxiv_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /arxiv <id> command."""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/arxiv <arxiv_id>`\nExample: `/arxiv 2303.08774`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    arxiv_id = context.args[0].strip()
    await send_typing(update, context)
    status_msg = await update.message.reply_text(f"⬇️ Fetching arXiv paper `{arxiv_id}`...", parse_mode=ParseMode.MARKDOWN)

    try:
        result, pdf_bytes = await research_engine.analyze_arxiv_paper(arxiv_id)
        await status_msg.delete()
        
        # Send PDF file if available
        if pdf_bytes:
            # Use BytesIO to send file from memory
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_file.name = f"{arxiv_id}.pdf"
            await update.message.reply_document(
                document=pdf_file,
                caption=f"📄 PDF for arXiv:{arxiv_id}"
            )

        await send_long_message(update, result)
    except Exception as e:
        logger.error(f"arXiv analysis error: {e}")
        await status_msg.edit_text(f"❌ Error analyzing paper: {e}")


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search <query> command."""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/search <query>`\nExample: `/search transformer attention mechanism`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    query = " ".join(context.args)
    await send_typing(update, context)
    status_msg = await update.message.reply_text(f"🔍 Searching for: `{query}`...", parse_mode=ParseMode.MARKDOWN)

    try:
        result = await research_engine.search_papers(query)
        await status_msg.delete()
        await send_long_message(update, result)
    except Exception as e:
        logger.error(f"Search error: {e}")
        await status_msg.edit_text(f"❌ Search failed: {e}")


async def litreview_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /litreview <topic> command."""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/litreview <topic>`\nExample: `/litreview diffusion models for image synthesis`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    topic = " ".join(context.args)
    await send_typing(update, context)
    status_msg = await update.message.reply_text(f"📚 Generating literature review for: `{topic}`...\n_This may take ~30 seconds_", parse_mode=ParseMode.MARKDOWN)

    try:
        result = await research_engine.generate_literature_review(topic)
        await status_msg.delete()
        await send_long_message(update, result)
    except Exception as e:
        logger.error(f"Lit review error: {e}")
        await status_msg.edit_text(f"❌ Error: {e}")


async def equation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /equation <eq> command."""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/equation <equation>`\nExample: `/equation L = -sum p(x) log q(x)`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    equation = " ".join(context.args)
    await send_typing(update, context)
    status_msg = await update.message.reply_text("📐 Analyzing equation...", parse_mode=ParseMode.MARKDOWN)

    try:
        result = await research_engine.explain_equations(equation)
        await status_msg.delete()
        await send_long_message(update, result)
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {e}")


async def datasets_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /datasets <task> command."""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/datasets <task>`\nExample: `/datasets sentiment analysis on financial news`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    task = " ".join(context.args)
    await send_typing(update, context)
    status_msg = await update.message.reply_text(f"📦 Finding datasets for: `{task}`...", parse_mode=ParseMode.MARKDOWN)

    try:
        result = await research_engine.recommend_datasets(task)
        await status_msg.delete()
        await send_long_message(update, result)
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {e}")


async def qa_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /qa <question> command."""
    if not context.args:
        await update.message.reply_text(
            "Usage: `/qa <question>`\nExample: `/qa What is the difference between BERT and GPT?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    question = " ".join(context.args)
    user_id = update.effective_user.id
    state = get_user_state(user_id)

    await send_typing(update, context)
    status_msg = await update.message.reply_text("🧠 Reasoning...", parse_mode=ParseMode.MARKDOWN)

    try:
        result = await research_engine.answer_research_question(
            question=question,
            chat_history=state["chat_history"][-6:],  # Last 3 turns
        )
        # Update chat history
        state["chat_history"].append({"role": "user", "content": question})
        state["chat_history"].append({"role": "assistant", "content": result[:500]})

        await status_msg.delete()
        await send_long_message(update, result)
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT MESSAGE HANDLER (Conversational Q&A fallback)
# ─────────────────────────────────────────────────────────────────────────────

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages as research Q&A."""
    text = update.message.text.strip()
    if not text or text.startswith("/"):
        return

    user_id = update.effective_user.id
    state = get_user_state(user_id)

    await send_typing(update, context)

    try:
        result = await research_engine.answer_research_question(
            question=text,
            chat_history=state["chat_history"][-6:],
        )
        state["chat_history"].append({"role": "user", "content": text})
        state["chat_history"].append({"role": "assistant", "content": result[:500]})
        await send_long_message(update, result)
    except Exception as e:
        logger.error(f"Text Q&A error: {e}")
        await update.message.reply_text(f"❌ Error processing your question: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK QUERY HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    # ── Menu callbacks: show instruction messages ──
    menu_instructions = {
        "menu_analyze": "📄 Send me a PDF file to analyze!",
        "menu_search": "🔍 Use: `/search <query>`\nExample: `/search vision transformers`",
        "menu_gaps": "🕳️ Send a PDF and I'll find the research gaps!\nOr use: `/gaps` after uploading.",
        "menu_experiments": "🧪 Send a PDF for experiment suggestions!",
        "menu_novelty": "💡 Send a PDF to generate novel research ideas!",
        "menu_litreview": "📚 Use: `/litreview <topic>`\nExample: `/litreview graph neural networks`",
        "menu_equations": "📐 Use: `/equation <equation>`\nExample: `/equation attention = softmax(QK^T/sqrt(d_k))V`",
        "menu_datasets": "📦 Use: `/datasets <task>`\nExample: `/datasets named entity recognition`",
        "menu_qa": "❓ Just type your research question!",
        "menu_arxiv": "🆔 Use: `/arxiv <id>`\nExample: `/arxiv 1706.03762`",
    }

    if data in menu_instructions:
        await query.edit_message_text(
            menu_instructions[data],
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # ── PDF analysis callbacks ──
    pdf_actions = ["pdf_analyze", "pdf_summary", "pdf_gaps", "pdf_experiments", "pdf_novelty"]
    if data in pdf_actions:
        file_id = context.user_data.get("last_pdf_file_id")
        if not file_id:
            await query.edit_message_text("❌ Session expired or file not found. Please re-upload the PDF.")
            return

        await query.edit_message_text("⏳ Processing... This may take 20-40 seconds.")

        try:
            pdf_bytes = await _download_pdf(context, file_id)

            if data == "pdf_analyze":
                result = await research_engine.analyze_paper_from_pdf(pdf_bytes)
            elif data == "pdf_summary":
                result = await research_engine.quick_summary(pdf_bytes)
            elif data == "pdf_gaps":
                result = await research_engine.detect_research_gaps(pdf_bytes)
            elif data == "pdf_experiments":
                result = await research_engine.suggest_experiments(pdf_bytes)
            elif data == "pdf_novelty":
                result = await research_engine.generate_novel_ideas(pdf_bytes)
            else:
                result = "❌ Unknown action."

            await query.delete_message()
            # Send result in chunks
            chat_id = update.effective_chat.id
            chunk_size = 4000
            chunks = [result[i : i + chunk_size] for i in range(0, len(result), chunk_size)]
            for chunk in chunks:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await context.bot.send_message(chat_id=chat_id, text=chunk)
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"PDF callback error: {e}")
            await query.edit_message_text(f"❌ Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log and report errors."""
    logger.error(f"Telegram error: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚠️ An unexpected error occurred. Please try again.\nIf the issue persists, use /clear to reset."
        )
