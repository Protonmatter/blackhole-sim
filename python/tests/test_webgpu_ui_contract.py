from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "web/webgpu/index.html"
MAIN_PATH = ROOT / "web/webgpu/src/main.js"


class _ContractParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.body_attrs = {}
        self.ids = {}
        self._open_ids = []
        self.text_by_id = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "body":
            self.body_attrs = attributes
        element_id = attributes.get("id")
        if element_id:
            self.ids[element_id] = {"tag": tag, **attributes}
            self.text_by_id.setdefault(element_id, [])
        self._open_ids.append(element_id)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self._open_ids:
            self._open_ids.pop()

    def handle_data(self, data):
        for element_id in self._open_ids:
            if element_id:
                self.text_by_id[element_id].append(data)


def _parse_index():
    html = INDEX_PATH.read_text(encoding="utf-8")
    parser = _ContractParser()
    parser.feed(html)
    return html, parser


def test_webgpu_entrypoint_defaults_to_cinematic_explore():
    _, parser = _parse_index()
    assert parser.body_attrs["data-ui"] == "explore"
    assert parser.ids["hud"]["tag"] == "header"
    assert parser.ids["status"]["role"] == "status"
    assert parser.ids["status"]["aria-live"] == "polite"
    assert parser.ids["status"]["aria-atomic"] == "true"


def test_webgpu_canvas_is_keyboard_focusable_and_described():
    _, parser = _parse_index()
    canvas_attrs = parser.ids["view"]
    assert canvas_attrs["tabindex"] == "0"
    assert "Drag" in canvas_attrs["aria-label"]
    assert "Space" in canvas_attrs["aria-label"]
    assert "wheel" in canvas_attrs["aria-label"].lower()
    assert "R" in canvas_attrs["aria-label"]


def test_lab_panel_has_accessible_hidden_contract():
    _, parser = _parse_index()
    assert parser.ids["labToggle"]["type"] == "button"
    assert parser.ids["labPanel"]["aria-hidden"] == "true"
    assert "inert" in parser.ids["labPanel"]


def test_explore_copy_preserves_scientific_boundary():
    html, parser = _parse_index()
    assert "Educational browser renderer" in html
    explore_text = " ".join(parser.text_by_id["hud"]).lower()
    assert "physics-equivalent" not in explore_text
    assert not re.search(r"\b(validated|accurate)\b", explore_text)


def test_main_wires_ui_mode_without_renderer_reload():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "resolveUiMode" in source
    assert "parseFilmConfig" in source
    assert "resolveInitialRunning" in source
    assert "from './ui_state.mjs'" in source
    assert "function setUiMode(" in source
    assert "history.replaceState" in source

    handler = re.search(
        r"labToggle\.addEventListener\('click',\s*\(\)\s*=>\s*\{(?P<body>.*?)\n\s*\}\);",
        source,
        re.DOTALL,
    )
    assert handler, "Lab toggle click handler is missing"
    assert "location.href" not in handler.group("body")
    assert "applyControlState" not in handler.group("body")


def test_hidden_lab_is_inert_and_escape_closes_it():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "labPanel.inert = mode !== 'lab'" in source
    assert "labPanel.setAttribute('aria-hidden', String(mode !== 'lab'))" in source
    assert "ev.key === 'Escape'" in source
    assert "setUiMode('explore')" in source


def test_reduced_motion_default_is_wired():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "matchMedia('(prefers-reduced-motion: reduce)')" in source
    assert "resolveInitialRunning(params" in source


def test_renderer_factories_return_frames_to_one_scheduler():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "async function createStokesRenderer()" in source
    assert "async function createFragmentRenderer()" in source
    assert "function startInteractiveLoop(activeRenderer)" in source
    assert source.count("renderFrame(timeMs") >= 2

    stokes_factory = source.split("async function createStokesRenderer()", 1)[1].split(
        "async function createFragmentRenderer()", 1
    )[0]
    fragment_factory = source.split("async function createFragmentRenderer()", 1)[1].split(
        "function installUiHandlers()", 1
    )[0]
    assert "requestAnimationFrame" not in stokes_factory
    assert "requestAnimationFrame" not in fragment_factory
    assert source.count("requestAnimationFrame(frame)") == 2


def test_main_exposes_deterministic_browser_capture_contract():
    source = MAIN_PATH.read_text(encoding="utf-8")
    for token in (
        "window.__blackhole",
        "state()",
        "renderAt(seconds)",
        "frame(type = 'image/png', quality = 0.95)",
        "device.queue.onSubmittedWorkDone()",
        "simulationState.forceFull = true",
        "canvasBlobToBase64",
    ):
        assert token in source


def test_film_mode_uses_exact_dimensions_and_readiness_markers():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "canvas.width = filmConfig.width" in source
    assert "canvas.height = filmConfig.height" in source
    assert "simulationState.timeSeconds = filmConfig.timeSeconds" in source
    assert "document.documentElement.dataset.blackholeReady = 'true'" in source
    assert "document.documentElement.dataset.blackholeError" in source


def test_film_resize_does_not_clear_an_unchanged_one_shot_frame():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert (
        "if (canvas.width !== filmConfig.width) canvas.width = filmConfig.width;"
        in source
    )
    assert (
        "if (canvas.height !== filmConfig.height) canvas.height = filmConfig.height;"
        in source
    )


def test_startup_failure_does_not_claim_readiness():
    source = MAIN_PATH.read_text(encoding="utf-8")
    html = INDEX_PATH.read_text(encoding="utf-8")
    assert "delete document.documentElement.dataset.blackholeReady" in source
    assert "delete document.documentElement.dataset.blackholeError" in source
    assert "catch (error)" in source
    assert 'html[data-blackhole-error] body[data-ui="film"] #hud' in html


def test_startup_validates_shader_pipeline_and_first_submission():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert ".getCompilationInfo()" in source
    assert "createComputePipelineAsync" in source
    assert "createRenderPipelineAsync" in source
    assert "device.pushErrorScope('validation')" in source
    assert "await device.popErrorScope()" in source
    assert "if (validationError)" in source


def test_ci_requires_dependency_free_web_tests():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "web:" in workflow
    assert "actions/setup-node@v4" in workflow
    assert 'node-version: "22"' in workflow
    assert (
        "node --test web/webgpu/tests/ui_state.test.mjs "
        "web/webgpu/tests/runtime_control.test.mjs"
    ) in workflow
    assert "needs: [hygiene, web, python, rust, science-contracts, native-wheel]" in workflow
