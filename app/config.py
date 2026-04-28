# app/config.py
CLASSIFIER_CKPT = "checkpoints/helpdesk-classifier"
KB_INDEX_DIR = "data/kb_index"
TOP_K = 6
TOP_K_RERANK = 3
WHITELIST_COMMANDS = ["ping", "ipconfig", "whoami", "echo"]