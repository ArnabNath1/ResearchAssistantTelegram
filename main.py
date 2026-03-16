"""
Main Entry Point — FastAPI + Telegram Bot (polling mode).
Launches both the FastAPI health/webhook API and the Telegram bot.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import sys

from config import settings
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from bot_handlers import (
    start_handler,
    help_handler,
    menu_handler,
    clear_handler,
    pdf_document_handler,
    arxiv_handler,
    search_handler,
    litreview_handler,
    equation_handler,
    datasets_handler,
    qa_handler,
    text_message_handler,
    callback_query_handler,
    error_handler,
)
from research_engine import research_engine


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP (Health, REST endpoints)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Assistant API",
    description="AI-powered research assistant backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class LitReviewRequest(BaseModel):
    topic: str


class DatasetRequest(BaseModel):
    task: str


class EquationRequest(BaseModel):
    equation: str
    context: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Research Assistant Bot"}


@app.post("/api/analyze-pdf")
async def api_analyze_pdf(file: UploadFile = File(...)):
    """Analyze a research PDF via REST API."""
    if file.content_type != "application/pdf":
        raise HTTPException(400, "File must be a PDF")
    pdf_bytes = await file.read()
    result = await research_engine.analyze_paper_from_pdf(pdf_bytes)
    return {"result": result}


@app.post("/api/question")
async def api_question(req: QuestionRequest):
    """Answer a research question via REST API."""
    result = await research_engine.answer_research_question(req.question)
    return {"result": result}


@app.post("/api/literature-review")
async def api_lit_review(req: LitReviewRequest):
    """Generate a literature review via REST API."""
    result = await research_engine.generate_literature_review(req.topic)
    return {"result": result}


@app.post("/api/datasets")
async def api_datasets(req: DatasetRequest):
    """Recommend datasets via REST API."""
    result = await research_engine.recommend_datasets(req.task)
    return {"result": result}


@app.post("/api/equation")
async def api_equation(req: EquationRequest):
    """Explain an equation via REST API."""
    result = await research_engine.explain_equations(req.equation, req.context)
    return {"result": result}


@app.get("/api/search")
async def api_search(q: str):
    """Search research papers via REST API."""
    result = await research_engine.search_papers(q)
    return {"result": result}


@app.get("/api/arxiv/{arxiv_id}")
async def api_arxiv(arxiv_id: str):
    """Analyze an arXiv paper by ID via REST API."""
    result, _ = await research_engine.analyze_arxiv_paper(arxiv_id)
    return {"result": result}


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM BOT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_telegram_app() -> Application:
    """Build and configure the Telegram bot application."""
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    # ── Command handlers ──
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("menu", menu_handler))
    application.add_handler(CommandHandler("clear", clear_handler))
    application.add_handler(CommandHandler("arxiv", arxiv_handler))
    application.add_handler(CommandHandler("search", search_handler))
    application.add_handler(CommandHandler("litreview", litreview_handler))
    application.add_handler(CommandHandler("equation", equation_handler))
    application.add_handler(CommandHandler("datasets", datasets_handler))
    application.add_handler(CommandHandler("qa", qa_handler))

    # Shorthand for single-word commands
    application.add_handler(CommandHandler("analyze", lambda u, c: u.message.reply_text("📄 Please send a PDF file to analyze!")))
    application.add_handler(CommandHandler("gaps", lambda u, c: u.message.reply_text("🕳️ Please send a PDF file first, then select 'Find Gaps'!")))
    application.add_handler(CommandHandler("experiments", lambda u, c: u.message.reply_text("🧪 Please send a PDF file first!")))
    application.add_handler(CommandHandler("novelty", lambda u, c: u.message.reply_text("💡 Please send a PDF file first!")))
    application.add_handler(CommandHandler("summary", lambda u, c: u.message.reply_text("📄 Please send a PDF file to summarize!")))

    # ── Document handler (PDF uploads) ──
    application.add_handler(MessageHandler(filters.Document.PDF, pdf_document_handler))

    # ── Inline keyboard callbacks ──
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    # ── Text messages → Research Q&A ──
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    # ── Error handler ──
    application.add_error_handler(error_handler)

    return application


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_telegram_bot():
    """Run Telegram bot in polling mode."""
    tg_app = build_telegram_app()
    logger.info("Starting Telegram bot (polling mode)...")
    await tg_app.initialize()
    await tg_app.start()
    await tg_app.updater.start_polling(drop_pending_updates=True)
    logger.info("✅ Telegram bot is running!")

    # Keep running until stopped
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()


async def run_all():
    """Run both FastAPI and Telegram bot concurrently."""
    # FastAPI server config
    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info(f"Starting FastAPI on {settings.api_host}:{settings.api_port}")

    # Run both concurrently
    await asyncio.gather(
        server.serve(),
        run_telegram_bot(),
    )


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/research_bot.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )

    logger.info("🚀 Launching Research Assistant Bot...")

    try:
        import gc
        gc.collect()
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
