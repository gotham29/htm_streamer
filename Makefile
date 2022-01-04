SHELL := /bin/bash

DIST_DIR = "dist"
BUILDT_DIR = "build"
DEPLOY_DIR = "deploy"
SHURI_ROOT = "shuri"

# User options
prefix ?= "tarball-shuri"
model_id ?= "ff9be7_hbt_prod_3"
version ?= "3.24-UAT"

PKG_NAME = "$(prefix)-$(version)"
PKG_PATH = "$(DIST_DIR)/$(PKG_NAME).tar"

fresh:
	rm -rf venv/
	virtualenv -p python venv
	. venv/bin/activate ;\
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	. venv/bin/activate ;\
	python -m pytest tests/
	python -m pytest tests/integration_tests.py

downloads:
	. venv/bin/activate ;\
	python -m nltk.downloader all

update:
	. venv/bin/activate ;\
	pip install -r requirements.txt

checkstyle:
	pep8 bom

pyenvbuild:
	pyenv shell 3.7.2
	pip install --upgrade pip
	pip install -r requirements.txt
	python -m pytest tests/

linting:
	pip install pylint
	pylint --max-line-length=120 --disable=C0114,R0913 src


# Build Shuri tarball
.PHONY: shuri
shuri:
	@if [[ -z "$(model_id)" ]]; then \
		echo "Option 'version' is required"; \
		exit 1; \
	fi
	@if [[ -z "$(version)" ]]; then \
		echo "Option 'version' is required"; \
		exit 1; \
	fi
	@if ! git diff --quiet; then \
		echo "Repo is dirty, please commit and try again"; \
		exit 1; \
	fi
	@git tag v$(version) && git push --tags
	@mkdir -p "$(DIST_DIR)" "$(BUILDT_DIR)"
	@rm -f "$(BUILDT_DIR)/$(PKG_NAME)" && \
		(cd "$(BUILDT_DIR)"; \
			rm -f "$(PKG_NAME)" && \
			cp -rf ../$(SHURI_ROOT) $(PKG_NAME) && \
			cp -rf ../$(DEPLOY_DIR) $(PKG_NAME)/$(model_id) && \
			find $(PKG_NAME) -name .DS_Store -delete && \
			tar -cvf $(PKG_NAME).tar $(PKG_NAME) && \
			mv $(PKG_NAME).tar ../$(DIST_DIR)/ \
		)
	@if [[ -f "$(PKG_PATH)" ]]; then \
		echo "Generated $(PKG_PATH)"; \
	else \
		echo "No tarball generated at $(PKG_PATH)"; \
		exit 1; \
	fi
