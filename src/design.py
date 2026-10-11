"""Tailwind-powered UI styling helpers for the Streamlit chatbot."""


import html

import streamlit as st


PAGE_TITLE = "Knowledge Base Chatbot"
PAGE_ICON = "💬"


_TAILWIND_CDN = '<script src="https://cdn.tailwindcss.com"></script>'


_CUSTOM_CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 860px;}
.stChatMessage {background: transparent;}
</style>
"""


def apply_design() -> None:
    """Inject Tailwind CSS and base styling, then render the page header."""
    st.markdown(_TAILWIND_CDN + _CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flex items-center gap-3 mb-6">
            <div class="flex items-center justify-center w-12 h-12 rounded-2xl
                        bg-gradient-to-br from-indigo-500 to-purple-600 text-2xl">
                🤖
            </div>
            <div>
                <h1 class="text-2xl font-bold text-slate-900 dark:text-white m-0">
                    Knowledge Base Chatbot
                </h1>
                <p class="text-sm text-slate-500 dark:text-slate-400 m-0">
                    Ask about accounts, projects, revenue, proposals, or the weather.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bubble(role: str, content: str) -> str:
    """Return a Tailwind-styled chat bubble for the given role/content."""

    safe_content = html.escape(content).replace("\n", "<br/>")
    is_user = role == "user"

    align = "justify-end" if is_user else "justify-start"

    bubble_style = (
        "bg-indigo-600 text-white rounded-br-sm"
        if is_user
        else "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 rounded-bl-sm"
    )

    return f"""
    <div class="flex {align} mb-3">
        <div class="max-w-[75%] px-4 py-2 rounded-2xl {bubble_style} shadow-sm">
            {safe_content}
        </div>
    </div>
    """
