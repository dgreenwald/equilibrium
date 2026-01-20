.PHONY: clean build check upload-test upload release-test release

clean:
	rm -rf dist build *.egg-info

build: clean
	python -m build

check: build
	python -m twine check dist/*

upload-test: check
	python -m twine upload --repository testpypi dist/*

upload: check
	python -m twine upload dist/*

release-test: upload-test
release: upload
