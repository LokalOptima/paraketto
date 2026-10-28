APP_NAME      := Paraketto
BUNDLE_ID     := io.lfrati.paraketto
BUNDLE        := $(APP_NAME).app
CONFIG        ?= release
BINARY        := .build/$(CONFIG)/$(APP_NAME)
CONTENTS      := $(BUNDLE)/Contents
# Uses the "Paraketto Dev" self-signed certificate when it exists (see README), otherwise
# ad-hoc. Ad-hoc signatures change on every build, which invalidates the Accessibility grant.
SIGN_IDENTITY ?= $(if $(shell security find-identity -p codesigning 2>/dev/null | grep -o '"Paraketto Dev"'),Paraketto Dev,-)

.PHONY: all build bundle run trust clean

all: bundle

build:
	swift build -c $(CONFIG)

bundle: build
	rm -rf $(BUNDLE)
	mkdir -p $(CONTENTS)/MacOS $(CONTENTS)/Resources
	cp $(BINARY) $(CONTENTS)/MacOS/$(APP_NAME)
	cp Resources/Info.plist $(CONTENTS)/Info.plist
	cp -R .build/$(CONFIG)/*.bundle $(CONTENTS)/Resources/ 2>/dev/null || true
	codesign --force --sign "$(SIGN_IDENTITY)" \
		--entitlements Resources/$(APP_NAME).entitlements \
		--options runtime $(BUNDLE)
	@echo "signed with: $(SIGN_IDENTITY)"

run: bundle
	open $(BUNDLE)

# Drops the stale Accessibility entry left by a previous (ad-hoc) build and opens the
# Accessibility pane so the running build can be granted again.
trust:
	tccutil reset Accessibility $(BUNDLE_ID)
	open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

clean:
	rm -rf $(BUNDLE) .build
