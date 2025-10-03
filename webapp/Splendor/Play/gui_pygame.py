# Splendor/Play/gui_pygame.py
"""Conducts the game and renderers through pygame."""

import sys
from pathlib import Path
from random import choice

import numpy as np
import pygame

from Splendor.Play import FocusTarget
from Splendor.Play.common_types import GUIMove
from Splendor.Play.render import (
    BoardGeometry,
    BoardRenderer,
    OverlayRenderer,
    Coord
)


def pil_to_surface(pil_image):
    img = pil_image.convert("RGB")  # force predictable mode
    return pygame.image.fromstring(img.tobytes(), img.size, "RGB")

class UILock:
    """Block all UI clicks while AI is thinking and
    after each human move to allow the player to see
    the consequences of their move.
    """
    def __init__(self, game, human):
        self._game = game
        self._human = human
        self.locked_until: int | None = None  # in ms

    @property
    def active(self) -> bool:
        now = pygame.time.get_ticks()
        human_pause = self.locked_until is not None and now < self.locked_until
        return human_pause or not self._human.awaiting_move

    def arm_delay(self, ms: int) -> None:
        self.locked_until = pygame.time.get_ticks() + ms


class SplendorGUI:
    def __init__(self, game, human):
        self.game = game
        self.human = human
        self.overlay: OverlayRenderer
        self._renderer = BoardRenderer()
        self.window = None
        self.running = True

        # UI locks
        self._awaiting_ai = False
        self.delay_after_move: int = 1600
        self.lock = UILock(game, human)

        # Caches
        self._preview_state: tuple[FocusTarget | None, tuple[int, ...]] = (None, ())
        self._preview_lines: list[str] = []

        # State
        self._focus_target: FocusTarget | None = None
        self._take_picks: list[int] = []
        self._take_discards: list[int] = []
        self._ctx_rects = {}  # maps overlay button to a move_idx
        self._spend_state = None  # Context for when player manually spends
        self._discard_state = False  # Flag for when reserving with 10 gems

        self._base_frame = None  # pygame.Surface
        self._scene_stamp = None  # int: game.half_turns of last base draw

        # Sound effects
        pygame.mixer.init(frequency=44100, channels=2, buffer=512)
        aud = Path(__file__).resolve().parent / "render" / "Resources" / "audio"
        self._sfx = {
            "coins": [pygame.mixer.Sound(str(p)) for p in (aud/"coins").glob("coin_*.ogg")],
            "cards": [pygame.mixer.Sound(str(p)) for p in (aud/"cards").glob("card_pull_*.ogg")]
        }
        for s in (self._sfx["coins"] + self._sfx["cards"]):
            s.set_volume(0.6)

    def _play_audio(self, key: str):
        if not getattr(self, "_mute_sfx", False):
            choice(self._sfx[key]).play()

    def _reset_overlay_inputs(self):
        self._focus_target = None
        self._take_picks.clear()
        self._take_discards.clear()
        self._ctx_rects.clear()
        self._spend_state = None
        self._discard_state = False

    def _is_gem_click_allowed(self, color: int) -> bool:
        """Click is allowed if 4 Splendor rules pass."""
        supply = self.game.board.gems[color]
        picked = self._take_picks

        # 1. Max three picks total (two for taking two of the same).
        if len(picked) >= 3:
            return False
        elif len(picked) == 2 and picked[0] == picked[1]:
            return False
        
        # 2. There must be at least one token of that kind.
        if supply == 0:
            return False
        
        # 3. A second click of the same color is allowed when:
        if len(picked) == 1 and picked[0] == color and supply < 4:
            return False
        
        # 4. Once you've picked 2 diff you must keep picking diff
        if len(picked) >= 2 and color in picked:
            return False
        
        # 5. A player can have at most 10 gems.
        # These clicks are allowed, but the Confirm button
        # will not enable until self.discards_required is 0.
        
        return True
    
    def _is_reserve_legal(self) -> bool:
        return len(self.game.active_player.reserved_cards) < 3

    def _reserve_needs_discard(self) -> bool:
        gold_exists = self.game.board.gems[5] > 0
        gem_capped = self.game.active_player.gems.sum() >= 10
        return gold_exists and gem_capped
    
    @property
    def discards_required(self) -> int:
        # Move this to the HumanAgent class?
        player = self.game.active_player
        n_picked = len(self._take_picks)
        n_discarded = len(self._take_discards)
        return max(0, player.gems.sum() + n_picked - n_discarded - 10)

    @staticmethod
    def compute_spend(card_cost, player) -> np.ndarray:
        card_cost = np.maximum(card_cost - player.cards, 0)
        spent_gems = np.minimum(player.gems, card_cost)
        remainder = card_cost.sum() - spent_gems.sum()
        spent_gems[5] = min(remainder, player.gems[5])
        return spent_gems
    
    def _handle_board_click(self, mouse_x, mouse_y, button: int) -> None:
        # Spend-selection mode; only accept player_gem clicks
        if self._spend_state or self._discard_state:
            for rect, token in reversed(list(self.clickmap.items())):
                if rect.contains(mouse_x, mouse_y):
                    if token[0] == "player_gem":
                        self._handle_spend_click(token[1], button)
                    break
            return

        # Normal mode
        for rect, token in reversed(list(self.clickmap.items())):
            if rect.contains(mouse_x, mouse_y):
                # Right click unfocuses any card
                if button == 3:
                    self._focus_target = None
                    if not token[0].endswith("gem"):
                        break
                
                # Clicking gems unfocuses cards and vice-versa
                if button == 1 and token[0].endswith("gem"):
                    self._focus_target = None
                elif button == 1 and token[0].endswith("card"):
                    self._take_picks.clear()
                    self._take_discards.clear()

                # Now apply click logic
                kind = token[0]
                if kind == "board_card":
                    self._focus_target = FocusTarget.from_index(*token[1:])
                elif kind == "reserved_card":
                    self._focus_target = FocusTarget("reserved", reserve_idx=token[1])
                elif kind == "board_gem":
                    # Add/remove gem based on l/r click
                    color = token[1]
                    if button == 3 and color in self._take_picks:
                        self._take_picks.remove(color)
                    elif button == 1 and self._is_gem_click_allowed(color):
                        self._take_picks.append(color)
                elif kind == "player_gem":
                    if self._spend_state is None and self.discards_required == 0 and button == 1:
                        break

                    # Add/remove gem based on l/r click
                    color = token[1]
                    if button == 3 and color in self._take_discards:
                        self._take_discards.remove(color)
                    elif button == 1:
                        if color in self._take_discards:
                            self._take_discards.remove(color)
                        else:
                            self._take_discards.append(color)

                break
    
    def _handle_context_menu_click(self, payload: tuple[str, GUIMove]) -> None:
        button_choice, move = payload

        if button_choice == "clear":
            ss = self._spend_state
            if ss is not None:
                if ss["spend"].sum():
                    ss["spend_picks"].clear()
                    ss["spend"][:] = 0
                else:
                    self._spend_state = None
            elif self._focus_target is not None:
                self._focus_target = None
                self._ctx_rects.clear()
            else:
                self._reset_overlay_inputs()
            return
        
        if button_choice == "confirm" and move is not None:
            if move.kind == "buy_choose" or move.kind == "reserve_with_discard":
                if move.kind == "reserve_with_discard":
                    self._discard_state = True
                self._start_spend_mode(move.source, move.card)
                return
            
            # Audio
            if move.kind == "take":
                self._play_audio("coins")
            else:
                self._play_audio("cards")
                if (move.kind == "reserve"
                    and self.game.active_player.gems.sum() < 10
                    and self.game.board.gems[5] > 0):
                    self._play_audio("coins")
            
            # Take, buy, reserve, or buy with chosen spend
            self.human.feed_move(move)
            self._reset_overlay_inputs()
            return

    def _handle_mouse_event(self, event: pygame.event.Event) -> None:
        mouse_x, mouse_y = event.pos
        sx, sy = self.overlay.scale()
        mouse_x = int(mouse_x / sx)
        mouse_y = int(mouse_y / sy)

        for rect, payload in self._ctx_rects.items():
            # Context menu buttons
            if rect.contains(mouse_x, mouse_y):
                self._handle_context_menu_click(payload)
                break
        else:
            # Regular board click
            self._handle_board_click(mouse_x, mouse_y, event.button)

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.lock.active:
                self._handle_mouse_event(event)
        elif event.type == pygame.USEREVENT and self._awaiting_ai:
            self._awaiting_ai = False
        elif event.type == pygame.QUIT:
            print("Exiting through pygame.QUIT")
            self.running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            print("Exiting through pygame.KEYDOWN")
            self.running = False

    def _card_menu_options(self, focus: FocusTarget) -> list[tuple[str, GUIMove]]:
        """Get legal (label, move) pairs for card clicks.
        We do this so we can draw the Confirm/Select button,
        and keep the legality checking to this one step.
        """

        p = self.game.active_player
        opts: list[tuple[str, GUIMove]] = []
        match focus.kind:
            case "shop":
                card = self.game.board.cards[focus.tier][focus.pos]
                afford_wo_gold, afford_with_gold = p.can_afford_card(card)
                if afford_wo_gold or afford_with_gold:
                    if afford_with_gold and p.gold_choice_exists(card):
                        # Allow for manually spending gems
                        move = GUIMove("buy_choose", card=card, source=focus)
                        opts.append(("Buy (mnl)", move))
                    
                    # Always offer an auto-spend option
                    spend = self.compute_spend(card.cost, p)
                    move = GUIMove("buy", card=card, source=focus, spend=spend)
                    opts.append(("Buy (auto)", move))

                if self._is_reserve_legal():
                    if self._reserve_needs_discard():
                        move = GUIMove("reserve_skip_gold", card=card, source=focus)
                        opts.append(("Reserve (skip gold)", move))

                        move = GUIMove("reserve_with_discard", card=card, source=focus)
                        opts.append(("Reserve (with discard)", move))
                    else:
                        move = GUIMove("reserve", card=card, source=focus)
                        opts.append(("Reserve", move))

            case "deck":
                if self._is_reserve_legal():
                    if self._reserve_needs_discard():
                        move = GUIMove("reserve_skip_gold", card=None, source=focus)
                        opts.append(("Reserve (skip gold)", move))

                        move = GUIMove("reserve_with_discard", card=None, source=focus)
                        opts.append(("Reserve (with discard)", move))
                    else:
                        move = GUIMove("reserve", card=None, source=focus)
                        opts.append(("Reserve", move))

            case "reserved":
                card = p.reserved_cards[focus.reserve_idx]
                afford_wo_gold, afford_with_gold = p.can_afford_card(card)
                if afford_wo_gold or afford_with_gold:
                    if afford_with_gold and p.gold_choice_exists(card):
                        # Allow for manually spending gems
                        move = GUIMove("buy_choose", card=card, source=focus)
                        opts.append(("Buy (mnl)", move))
                    
                    # Always offer an auto-spend option
                    spend = self.compute_spend(card.cost, p)
                    move = GUIMove("buy", card=card, source=focus, spend=spend)
                    opts.append(("Buy (auto)", move))

        return opts

    def _start_spend_mode(self, focus, card) -> None:
        """Engaged for self._spend_state or self._discard_state."""
        p = self.game.active_player

        # Variables limiting what can be discarded
        if self._discard_state:
            cost = np.zeros(6, dtype=int)
            colored_max = p.gems[:5].astype(int)
            gold_max = int(p.gems[5])
        else:
            cost = np.maximum(card.cost - p.cards, 0)
            colored_max = np.minimum(p.gems[:5], cost[:5]).astype(int)
            gold_max = int(min(p.gems[5], cost[:5].sum()))

        self._spend_state = {
            "focus": focus,
            "card": card,
            "cost": cost,
            "colored_max": colored_max,
            "gold_max": gold_max,
            "spend_picks": [],
            "spend": np.zeros(6, dtype=int),
        }

        self._take_picks.clear()
        self._take_discards.clear()
        self._focus_target = focus  # keep the yellow outline on the card

    def _handle_spend_click(self, color: int, button: int) -> None:
        ss = self._spend_state
        if not ss or button not in (1,3):
            return

        picks = ss["spend_picks"]
        need = 1 if self._discard_state else ss["cost"][:5].sum()
        used = np.bincount(picks, minlength=6)[:6]
        total = used[:5].sum() + used[5]
        cap = ss["colored_max"][color] if color < 5 else ss["gold_max"]

        if button == 1:
            # Add one if under the need
            if used[color] < cap and total < need:
                picks.append(color)
        else:
            # Remove one on right click
            for i in range(len(picks) - 1, -1, -1):
                if picks[i] == color:
                    picks.pop(i)
                    break

        ss["spend"][:] = np.bincount(picks, minlength=6)[:6]

    def tick(self) -> None:
        """Process one frame."""
        if not self.running:
            return

        # Outer main() sets up pygame/init/window.
        if self.window is None:
            self.window = pygame.display.get_surface()
            if self.window is None:
                return

        # Lazily create overlay once we have a window.
        if not hasattr(self, "overlay"):
            self.overlay = OverlayRenderer(self.window)
            pygame.display.set_caption("Splendor RL - Human vs DDQN")
    
        # 1) Ensure we have a fresh clickmap and cached base frame
        scene_stamp = self.game.half_turns
        if self._scene_stamp != scene_stamp or self._base_frame is None:
            self.clickmap, pil_img = self._renderer.render(self.game)
            pil_img = pil_img.convert("RGB")
            self._base_frame = pygame.image.frombuffer(
                pil_img.tobytes(), pil_img.size, "RGB"
            )
            self._scene_stamp = scene_stamp

        # 2) Now poll events using the current clickmap
        for event in pygame.event.get():
            self._handle_event(event)

        # 3) Blit cached base frame
        self.window.blit(self._base_frame, (0, 0))

        # Victory banner
        if self.game.victor:
            font = pygame.font.SysFont(None, 72)
            msg = "You win!" if any(p.victor and p.agent is self.human for p in self.game.players) else "You lose!"
            txt  = font.render(msg, True, (255, 215, 0))
            rect = txt.get_rect(center=self.window.get_rect().center)
            self.window.blit(txt, rect)
            pygame.display.flip()
            self.running = False
            return

        # Skip further logic when display is locked
        if self.lock.active:
            pygame.display.flip()
            return
    
        # Draw UI overlays
        self.overlay.draw_selection_highlights(
            self.clickmap, self._focus_target,
            self._take_picks, self._take_discards,
            self._spend_state["spend"] if self._spend_state else [0]*6
        )
        if self.discards_required > 0:
            self.overlay.draw_discard_notice()
        if (self._focus_target is not None) and self._reserve_needs_discard():
            self.overlay.draw_discard_notice(where="shop")

        ss = self._spend_state
        if ss:
            # Draw confirm when selected gems equals the cost
            if self._discard_state:
                need = 1
                cur = ss["spend"][:5].sum() + ss["spend"][5]
                confirm_enabled = (cur == need)
                move = GUIMove(
                    kind="reserve",  # Engine should treat reserve+discard properly
                    card=ss["card"],
                    source=ss["focus"],
                    spend=None,
                    discard=ss["spend"].copy(),
                )
            else:
                need = ss["cost"][:5].sum()
                cur = ss["spend"][:5].sum() + ss["spend"][5]
                confirm_enabled = (cur == need)
                move = GUIMove(
                    kind="buy",
                    card=ss["card"],
                    source=ss["focus"],
                    spend=ss["spend"].copy(),
                )

            self._ctx_rects = self.overlay.draw_move_confirm_button(
                move, confirm_enabled, clear_enabled=True
            )
        elif self._focus_target:
            # Draw Submit/Clear button and context menu for clicked cards
            options = self._card_menu_options(self._focus_target)
            origin = None

            match self._focus_target.kind:
                case "shop":
                    origin = next(
                        (Coord(r.x0, r.y0) for r, p in self.clickmap.items()
                         if p == ("board_card",
                                  self._focus_target.tier,
                                  self._focus_target.pos)),
                        None,
                    )
                case "deck":
                    origin = next(
                        (Coord(r.x0, r.y0) for r, p in self.clickmap.items()
                         if p == ("board_card", self._focus_target.tier, 4)),
                        None,
                    )
                case "reserved":
                    origin = next(
                        (Coord(r.x0, r.y0) for r, p in self.clickmap.items()
                         if p[0] == "reserved_card"
                         and p[1] == self._focus_target.reserve_idx),
                        None,
                    )

            if origin is not None and options:
                self._ctx_rects = self.overlay.draw_card_context_menu(
                    origin, options,
                )
        elif self._take_picks:
            # Draw Submit/Clear button for clicked tokens
            move = GUIMove(
                kind="take",
                take=np.bincount(self._take_picks, minlength=6)[:6],
                discard=np.bincount(self._take_discards, minlength=6)[:6]
            )
            confirm_enabled = (self.discards_required == 0)
            clear_enabled = bool(self._take_picks)
            self._ctx_rects = self.overlay.draw_move_confirm_button(
                move, confirm_enabled, clear_enabled
            )
        else:
            self._ctx_rects.clear()

        pygame.display.flip()

    def run(self):
        """Desktop wrapper that runs one non-blocking frame per iteration."""
        pygame.init()
        self.window = pygame.display.set_mode(
            BoardGeometry().canvas,
            pygame.RESIZABLE
        )
        self.overlay = OverlayRenderer(self.window)
        pygame.display.set_caption("Splendor vs AI")

        clock = pygame.time.Clock()
        while self.running:
            self.tick()
            clock.tick(60)

        pygame.quit()
        sys.exit()
