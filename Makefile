.PHONY: test eval validate

test:
	python3 -m unittest

eval:
	python3 -m prfsearch.evaluate

validate: test eval
