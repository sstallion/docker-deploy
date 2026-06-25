FROM vllm/vllm-openai:v0.22.1

RUN apt-get update && apt-get -y install \
  git \
  && rm -rf /var/lib/apt/lists/*

# Patch installation to support Qwen3 Intel AutoRound quants:
RUN curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/35156.diff | \
  git apply -R --directory=/usr/local/lib/python3.12/dist-packages --unsafe-paths

# Clean up support packages:
RUN apt-get -y remove --autoremove --purge \
  git
