test: 
	pytest --verbose test_myclassifiers.py -vv
lint:
	pylint -v $(shell git ls-files "*.py")
fix:
	autopep8 --in-place --aggressive --aggressive $(shell git ls-files "*.py")