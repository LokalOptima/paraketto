APP_NAME      := Paraketto
BUNDLE_ID     := io.lfrati.paraketto
BUNDLE        := $(APP_NAME).app
CONFIG        ?= release
BINARY        := .build/$(CONFIG)/$(APP_NAME)
CLI           := .build/$(CONFIG)/paraketto-cli
CONTENTS      := $(BUNDLE)/Contents
VAD_PACKAGE   := Resources/VadHead.mlpackage
VAD_COMPILED  := .build/VadHead.mlmodelc
FIXTURES      := tests/fixtures/librispeech_fluidaudio.tsv
CLIPS         ?= ../paraketto/data/librispeech/*.wav
# Signing. macOS ties the Accessibility grant to the app's signature. With the "Paraketto Dev"
# self-signed certificate in the login keychain (see README) every build carries the same
# signature and the grant survives; without it codesign falls back to ad-hoc ("-"), whose
# signature changes on every build. The certificate is selected by the SHA-1 hash that
# `security find-identity` prints, so a certificate imported twice is not "ambiguous" to codesign.
# Override with SIGN_IDENTITY=<hash or name>; the bundle target then checks the signature
# against SIGN_NAME, so set that too for a differently named certificate.
SIGN_NAME     ?= Paraketto Dev
SIGN_IDENTITY ?= $(or $(shell security find-identity -p codesigning 2>/dev/null | grep -F '"$(SIGN_NAME)"' | head -1 | awk '{print $$2}'),-)
SIGN_IDENTITY := $(SIGN_IDENTITY)

.PHONY: all build bundle run trust vad cli compare vad-export encoder-export decoder-export clean

MODELS_DIR    := $(HOME)/Library/Application Support/Paraketto/models

all: bundle

build:
	swift build -c $(CONFIG)

# Compiles the VAD head once; the .mlpackage in Resources is the source of truth.
vad: $(VAD_COMPILED)

$(VAD_COMPILED): $(VAD_PACKAGE)
	rm -rf $(VAD_COMPILED)
	xcrun coremlcompiler compile $(VAD_PACKAGE) .build

bundle: build vad
	rm -rf $(BUNDLE)
	mkdir -p $(CONTENTS)/MacOS $(CONTENTS)/Resources
	cp $(BINARY) $(CONTENTS)/MacOS/$(APP_NAME)
	cp Resources/Info.plist $(CONTENTS)/Info.plist
	cp -R $(VAD_COMPILED) $(CONTENTS)/Resources/
# The first codesign with a new key blocks on a keychain dialog until "Always Allow" is clicked.
	codesign --force --sign "$(SIGN_IDENTITY)" \
		--entitlements Resources/$(APP_NAME).entitlements \
		--options runtime $(BUNDLE)
	codesign --verify --deep --strict $(BUNDLE)
ifneq ($(SIGN_IDENTITY),-)
# An ad-hoc build would look fine but lose the Accessibility grant, so refuse to keep one.
	@codesign -dvv $(BUNDLE) 2>&1 | grep -q '^Authority=$(SIGN_NAME)$$' \
		|| { echo "error: $(BUNDLE) is not signed by $(SIGN_NAME); removing it"; rm -rf $(BUNDLE); exit 1; }
	@echo "signed with: $(SIGN_NAME) ($(SIGN_IDENTITY))"
else
	@echo "signed ad-hoc: no $(SIGN_NAME) certificate found, the Accessibility grant will not survive rebuilds"
endif

run: bundle
	open $(BUNDLE)

# Drops the stale Accessibility entry left by a previous (ad-hoc) build and opens the
# Accessibility pane so the running build can be granted again.
trust:
	tccutil reset Accessibility $(BUNDLE_ID)
	open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

cli: build vad

# Transcribes the LibriSpeech clips and checks every transcript against FluidAudio's.
compare: cli
	$(CLI) --vad $(VAD_COMPILED) --compare $(FIXTURES) $(CLIPS)

# Re-exports the VAD head from moondream/parakeet-ultra (needs uv; fetches ~9 MB of weights).
vad-export:
	cd tools/export && uv run python export_vad_head.py --out ../../$(VAD_PACKAGE)

# Exports the encoder with 5 s, 10 s and 15 s windows (needs uv; downloads the 1.2 GB checkpoint once)
# and compiles it where the app and CLI pick it up automatically.
encoder-export:
	cd tools/export && uv run python export_encoder.py --out "$(MODELS_DIR)/parakeet-ultra-encoder/Encoder.mlpackage"
	rm -rf "$(MODELS_DIR)/parakeet-ultra-encoder/Encoder.mlmodelc"
	xcrun coremlcompiler compile "$(MODELS_DIR)/parakeet-ultra-encoder/Encoder.mlpackage" "$(MODELS_DIR)/parakeet-ultra-encoder"

# Exports the fused decoder+joint step model (needs uv; fetches ~36 MB of weights) and compiles it.
decoder-export:
	cd tools/export && uv run python export_fused_decoder.py --out "$(MODELS_DIR)/parakeet-ultra-decoder/DecoderJoint.mlpackage"
	rm -rf "$(MODELS_DIR)/parakeet-ultra-decoder/DecoderJoint.mlmodelc"
	xcrun coremlcompiler compile "$(MODELS_DIR)/parakeet-ultra-decoder/DecoderJoint.mlpackage" "$(MODELS_DIR)/parakeet-ultra-decoder"

clean:
	rm -rf $(BUNDLE) .build
