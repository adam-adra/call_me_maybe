# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: adadra <adadra@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/04/27 23:30:48 by adadra            #+#    #+#              #
#    Updated: 2026/05/14 00:28:43 by adadra           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #



PYTHON = python3
MODULE = src

UV = uv

RUN_CMD = $(UV) run $(PYTHON) -m $(MODULE)

install:
	$(UV) sync

run:
	$(RUN_CMD)

debug:
	$(UV) run $(PYTHON) -m pdb -m $(MODULE)

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: install run debug lint lint-strict clean