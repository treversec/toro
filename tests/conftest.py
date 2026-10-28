# Silence Streamlit's "missing ScriptRunContext" chatter when tests exercise
# session-state-backed render functions outside `streamlit run`.
import logging

logging.disable(logging.CRITICAL)
