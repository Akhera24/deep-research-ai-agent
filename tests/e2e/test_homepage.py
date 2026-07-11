"""
Browser tests for the Phase D1 homepage: entity card carousel + rotating
headline name (PLAN.md Rev 3.2, D1/D5).

Covers: cards render with the curated stat chips; click pre-fills the search
input; loop clones are a11y-inert; the carousel drifts and pauses on hover;
the rotating name cycles and mirrors into the placeholder; reduced-motion
renders static; no-JS falls back to the static text strip.
"""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

VISIBLE_CARDS = ".ent-card:not([aria-hidden])"


class TestEntityCards:
    def test_cards_render_with_curated_stat_chips(self, server, page):
        page.goto(server)
        cards = page.locator(VISIBLE_CARDS)
        expect(cards).to_have_count(14)
        first = cards.first
        expect(first).to_contain_text("Jensen Huang")
        expect(first).to_contain_text("CEO · Semiconductors")
        expect(first).to_contain_text("97 facts")
        expect(first).to_contain_text("Grade A+")
        # Stripe's grade chip appeared once a measured run reached A-range
        # (decision ①: real numbers only — 10-iteration run scored 90.6/A-).
        stripe = page.locator(VISIBLE_CARDS, has_text="Stripe").first
        expect(stripe).to_contain_text("108 facts")
        expect(stripe).to_contain_text("Grade A-")

    def test_loop_clones_are_inert_for_a11y(self, server, page):
        page.goto(server)
        clones = page.locator(".ent-card[aria-hidden='true']")
        expect(clones).to_have_count(14)
        assert clones.first.get_attribute("tabindex") == "-1"

    def test_card_click_prefills_search_input(self, server, page):
        page.goto(server)
        page.hover("#entity-scroll")          # pause the drift for a stable click
        page.locator(VISIBLE_CARDS, has_text="Satya Nadella").first.click()
        expect(page.locator("#query")).to_have_value("Satya Nadella")
        expect(page.locator("#query")).to_be_focused()

    def test_carousel_drifts_and_pauses_on_hover(self, server, page):
        page.goto(server)
        scroll = page.locator("#entity-scroll")
        page.wait_for_timeout(600)
        s1 = scroll.evaluate("el => el.scrollLeft")
        page.wait_for_timeout(600)
        s2 = scroll.evaluate("el => el.scrollLeft")
        assert s2 > s1, "carousel should auto-drift"
        page.hover("#entity-scroll")
        p1 = scroll.evaluate("el => el.scrollLeft")
        page.wait_for_timeout(600)
        p2 = scroll.evaluate("el => el.scrollLeft")
        assert p2 == p1, "drift must pause on hover"


class TestRotatingName:
    def test_name_rotates_and_mirrors_placeholder(self, server, page):
        page.goto(server)
        # First tick is deterministic: entity[1] follows the static default.
        expect(page.locator("#query")).to_have_attribute(
            "placeholder", "e.g. Tim Cook", timeout=5000)
        expect(page.locator("#rot-name")).to_have_text("Tim Cook", timeout=5000)


class TestMotionFallbacks:
    def test_reduced_motion_renders_static(self, server, browser):
        ctx = browser.new_context(reduced_motion="reduce")
        page = ctx.new_page()
        try:
            page.goto(server)
            page.wait_for_timeout(3200)   # > one 2.5 s rotation period
            expect(page.locator("#rot-name")).to_have_text("Jensen Huang")
            assert page.locator("#entity-scroll").evaluate(
                "el => el.scrollLeft") == 0, "no drift under reduced motion"
        finally:
            ctx.close()

    def test_no_js_falls_back_to_static_strip(self, server, browser):
        ctx = browser.new_context(java_script_enabled=False)
        page = ctx.new_page()
        try:
            page.goto(server)
            expect(page.locator("#rot-name")).to_have_text("Jensen Huang")
            expect(page.locator(".ent-fallback")).to_be_visible()
            expect(page.locator("#entity-scroll")).to_be_hidden()
            expect(page.locator("#demo")).to_be_hidden()
            expect(page.locator("#quote")).to_contain_text("Francis Bacon")
        finally:
            ctx.close()


class TestHeroDemo:
    def test_demo_types_feeds_and_reveals_report_card(self, server, page):
        page.goto(server)
        expect(page.locator("#demo-query")).to_have_text("Jensen Huang",
                                                         timeout=4000)
        expect(page.locator("#demo-feed li").first).to_be_visible(timeout=4000)
        expect(page.locator("#demo-report")).to_be_visible(timeout=13000)
        expect(page.locator("#demo-report")).to_contain_text("97.3")

    def test_demo_pauses_on_hover(self, server, page):
        page.goto(server)
        expect(page.locator("#demo-feed li").first).to_be_visible(timeout=5000)
        page.hover("#demo")
        n1 = page.locator("#demo-feed li").count()
        page.wait_for_timeout(1600)   # > one step interval while paused
        assert page.locator("#demo-feed li").count() == n1

    def test_demo_reduced_motion_shows_static_final_frame(self, server, browser):
        ctx = browser.new_context(reduced_motion="reduce")
        page = ctx.new_page()
        try:
            page.goto(server)
            # No waiting: the final frame renders immediately, no loop.
            expect(page.locator("#demo-query")).to_have_text("Jensen Huang",
                                                             timeout=2000)
            expect(page.locator("#demo-report")).to_be_visible(timeout=2000)
        finally:
            ctx.close()


class TestQuoteAndSections:
    def test_quote_rotates(self, server, page):
        page.goto(server)
        expect(page.locator("#quote")).not_to_contain_text("Francis Bacon",
                                                           timeout=9000)

    def test_report_screenshots_load_from_static_mount(self, server, page):
        page.goto(server)
        page.locator("#shots").scroll_into_view_if_needed()
        first = page.locator("#shots-person img").first
        expect(first).to_be_visible()
        page.wait_for_timeout(500)
        assert first.evaluate("img => img.naturalWidth") == 1000

    def test_sample_cta_opens_both_reports_in_new_tabs(self, server, page):
        page.goto(server)
        for link_text, subject in [("person", "Jensen Huang"),
                                   ("company", "Stripe")]:
            with page.context.expect_page() as popup:
                page.locator("#sample-cta a", has_text=link_text).click()
            report = popup.value
            expect(report.locator("h2").first).to_contain_text(subject)
            expect(report.locator(".home-link")).to_have_attribute("href", "/")
            report.close()
        # Original tab kept its state — the 3-runs/hour flow depends on it.
        expect(page.locator("#query")).to_be_visible()

    def test_screenshot_click_deep_links_into_live_report(self, server, page):
        page.goto(server)
        risk_link = page.locator('#shots-person a[href*="risksSection"]')
        risk_link.scroll_into_view_if_needed()
        with page.context.expect_page() as popup:
            risk_link.click()
        report = popup.value
        assert report.url.endswith("/sample-report/person#risksSection")
        expect(report.locator("#risksSection")).to_be_visible()
        # The anchor target actually scrolled into view (deep link worked).
        in_view = report.locator("#risksSection").evaluate(
            "el => el.getBoundingClientRect().top < window.innerHeight")
        assert in_view


class TestShotsToggle:
    def test_defaults_to_person_set(self, server, page):
        page.goto(server)
        expect(page.locator("#shots-person")).to_be_visible()
        expect(page.locator("#shots-company")).to_be_hidden()
        expect(page.locator("#shots-tab-person")).to_have_attribute(
            "aria-selected", "true")

    def test_toggle_swaps_sets_and_survives_repeats(self, server, page):
        page.goto(server)
        page.locator("#shots").scroll_into_view_if_needed()
        # person -> company -> person -> company: state must stay coherent
        for _ in range(2):
            page.click("#shots-tab-company")
            expect(page.locator("#shots-company")).to_be_visible()
            expect(page.locator("#shots-person")).to_be_hidden()
            page.click("#shots-tab-person")
            expect(page.locator("#shots-person")).to_be_visible()
            expect(page.locator("#shots-company")).to_be_hidden()
        # After all toggling, hrefs are still the baked-in correct ones.
        assert page.locator(
            '#shots-person a[href="/sample-report/person#factsSection"]').count() == 1
        assert page.locator(
            '#shots-company a[href="/sample-report/company#factsSection"]').count() == 1

    def test_company_images_load_after_toggle(self, server, page):
        page.goto(server)
        page.locator("#shots").scroll_into_view_if_needed()
        page.click("#shots-tab-company")
        first = page.locator("#shots-company img").first
        expect(first).to_be_visible()
        page.wait_for_timeout(600)   # lazy images fetch once revealed
        assert first.evaluate("img => img.naturalWidth") == 1000

    def test_company_deep_link_opens_stripe_report_at_section(self, server, page):
        page.goto(server)
        page.locator("#shots").scroll_into_view_if_needed()
        page.click("#shots-tab-company")
        link = page.locator('#shots-company a[href*="risksSection"]')
        with page.context.expect_page() as popup:
            link.click()
        report = popup.value
        assert report.url.endswith("/sample-report/company#risksSection")
        expect(report.locator("h2").first).to_contain_text("Stripe")
        in_view = report.locator("#risksSection").evaluate(
            "el => el.getBoundingClientRect().top < window.innerHeight")
        assert in_view

    def test_toggle_works_under_reduced_motion(self, server, browser):
        ctx = browser.new_context(reduced_motion="reduce")
        page = ctx.new_page()
        try:
            page.goto(server)
            page.locator("#shots").scroll_into_view_if_needed()
            page.click("#shots-tab-company")
            expect(page.locator("#shots-company")).to_be_visible()
            expect(page.locator("#shots-person")).to_be_hidden()
        finally:
            ctx.close()


class TestPerfBudgets:
    def test_cls_under_budget(self, server, page):
        """D5 gate: CLS < 0.1, measured through rotations, demo steps and a
        full scroll (lazy sections included)."""
        page.goto(server)
        page.evaluate("""() => {
            window.__cls = 0;
            new PerformanceObserver(list => {
                for (const e of list.getEntries())
                    if (!e.hadRecentInput) window.__cls += e.value;
            }).observe({type: 'layout-shift', buffered: true});
        }""")
        page.wait_for_timeout(3000)               # name/quote/demo cycles
        page.mouse.wheel(0, 4000)                 # reveal lazy sections
        page.wait_for_timeout(1500)
        cls = page.evaluate("window.__cls")
        assert cls < 0.1, f"CLS {cls} exceeds 0.1"

    def test_initial_first_party_transfer_under_400kb(self, server, browser):
        """D5 gate: initial page weight (first-party, lazy assets excluded —
        no scrolling happens here) < 400 KB."""
        ctx = browser.new_context()
        page = ctx.new_page()
        sizes = []

        def track(response):
            if response.url.startswith(server):
                try:
                    sizes.append((response.url, len(response.body())))
                except Exception:
                    pass

        page.on("response", track)
        try:
            # NOT networkidle: Turnstile's external challenge polls forever.
            page.goto(server, wait_until="load")
            page.wait_for_timeout(1000)
            total = sum(s for _, s in sizes)
            assert total < 400 * 1024, (
                f"initial transfer {total/1024:.0f} KB: "
                + ", ".join(f"{u.rsplit('/', 1)[-1]}={s//1024}KB" for u, s in sizes))
        finally:
            ctx.close()
