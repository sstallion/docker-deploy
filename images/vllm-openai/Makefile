DOCKER		= docker
DOCKER_FLAGS	= --platform linux/arm64
DOCKER_IMAGE   	= $(shell awk '/^FROM/ { print $$NF }' Dockerfile)
DOCKER_REGISTRY	= registry.quuxotic.net
DOCKER_TAG	= $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)

.PHONY: all
all: build push

.PHONY: build
build:
	$(DOCKER) build $(DOCKER_FLAGS) -t $(DOCKER_TAG) .

.PHONY: push
push:
	$(DOCKER) push $(DOCKER_TAG)
