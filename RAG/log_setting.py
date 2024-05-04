import logging, os, sys
from loguru import logger


class InterceptHandler(logging.Handler):
	def emit(self, record):
		# Get corresponding Loguru level if it exists
		try:
			level = logger.level(record.levelname).name
		except ValueError:
			level = record.levelno

		# Find caller from where originated the logged message
		frame, depth = logging.currentframe(), 2
		while frame.f_code.co_filename == logging.__file__:
			frame = frame.f_back
			depth += 1

		logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logger() -> None:
	logging.getLogger().handlers = [InterceptHandler()]
	# logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG if settings.DEBUG else logging.NOTSET)
	logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
	logger.remove()
	logger.add(sys.stdout, colorize=True, level="INFO")
	logger.add(
		os.path.join('logs', "doc_ai.log"),
		rotation="1 days",
		level="INFO",
		retention="12 months",
		encoding='UTF-8'
	)