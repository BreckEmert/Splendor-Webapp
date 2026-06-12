# docs/main.py
# /// script
# dependencies = ["numpy", "pygame-ce", "Pillow"]
# ///

import asyncio
import sys, os
from importlib.resources import files
from pathlib import Path

import pygame
from js import console  # type: ignore


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
os.chdir(APP_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = console.log
log(f"[boot] cwd={os.getcwd()}")
log(f"[boot] sys.path[0:5]={sys.path[:5]}")

try:
    from Splendor.Environment.gui_game import GUIGame
    from Splendor.Environment.Splendor_components.Player_components import HumanAgent
    from Splendor.Play.gui_pygame import SplendorGUI
    from Splendor.Play.render import BoardGeometry
    from Splendor.RL import InferenceModel
    from Splendor.RL.search_model import SearchModel
    log("[boot] game modules imported OK")
except Exception:
    import traceback
    from js import console  # type: ignore
    console.error("[boot] import crash:\n" + traceback.format_exc())
    raise


def _resolve_model_path() -> str:
    env = os.getenv("MODEL_PATH")
    if env:
        return env
    return str(files("Splendor.RL.trained_agents") / "inference_model.npz")


def _resolve_value_path() -> str:
    return str(files("Splendor.RL.trained_agents") / "inference_critic.npz")


# Difficulty = how many search simulations the AI runs per move
# (0 = plain net move).
DIFFICULTIES = [
    ("Easy", 0),
    ("Medium", 50),
    ("Hard", 400),
]


async def choose_difficulty(clock) -> int:
    """Pre-game menu: three buttons, returns the chosen sims count."""
    surf = pygame.display.get_surface()
    W, H = surf.get_size()
    title_font = pygame.font.SysFont(None, int(H * 0.07))
    btn_font = pygame.font.SysFont(None, int(H * 0.045))

    bw, bh, gap = int(W * 0.22), int(H * 0.14), int(W * 0.04)
    total = 3 * bw + 2 * gap
    x0, y0 = (W - total) // 2, int(H * 0.45)
    rects = [pygame.Rect(x0 + i * (bw + gap), y0, bw, bh) for i in range(3)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(rects):
                    if r.collidepoint(event.pos):
                        return DIFFICULTIES[i][1]

        surf.fill((24, 28, 36))
        t = title_font.render("Choose difficulty", True, (235, 235, 235))
        surf.blit(t, t.get_rect(center=(W // 2, int(H * 0.3))))
        for i, r in enumerate(rects):
            name = DIFFICULTIES[i][0]
            hover = r.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, (70, 110, 90) if hover else (52, 64, 84),
                             r, border_radius=10)
            n = btn_font.render(name, True, (240, 240, 240))
            surf.blit(n, n.get_rect(center=r.center))
        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)


async def render_pause(ms, gui, clock):
    end = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < end:
        gui.tick()  # draw updated board
        clock.tick(60)  # 60 FPS
        await asyncio.sleep(0)  # yield to the browser/event loop

async def main():
    # Set up pygame
    pygame.init()
    pygame.font.init()

    pygame.display.set_mode(BoardGeometry().canvas)
    surf = pygame.display.get_surface()
    log(f"[pygame] set_mode OK, surface={surf.get_size()}")
    clock = pygame.time.Clock()

    # Difficulty menu (sims per AI move; 0 = plain net, instant)
    sims = await choose_difficulty(clock)
    log(f"[boot] difficulty selected: {sims} sims/move")

    # Agents + game (policy net + value net; searches when sims > 0)
    model_path = _resolve_model_path()
    log(f"[boot] loading model: {model_path}")
    rl_agent = SearchModel(model_path, _resolve_value_path(), sims=sims)
    human = HumanAgent()
    players = [("DDQN", rl_agent, 0), ("Human", human, 1)]

    game = GUIGame(players, rl_agent)
    rl_agent.bind(game)
    gui = SplendorGUI(game, human)

    running = True
    while running:
        try:
            # 1) draw + poll events (can queue a human move)
            gui.tick()

            # 2) unlock human input
            human_turn = game.active_player.name == "Human"
            human_move_ready = not human._move_queue.empty()
            if human_turn and not human_move_ready and not human.awaiting_move:
                human.awaiting_move = True

            # 3) advance game
            if human_move_ready:
                game.turn()
                await render_pause(gui.delay_after_move, gui, clock)
            elif not human_turn:
                game.turn()

        except Exception:
            import traceback
            console.error(traceback.format_exc())
            running = False

        running &= bool(getattr(gui, "running", True))
        clock.tick(60)
        await asyncio.sleep(0)

    pygame.quit()


if __name__ == "__main__":
    import traceback
    try:
        console.log("[boot] asyncio.run(main())")
        asyncio.run(main())
        console.log("[boot] main() finished")
    except Exception:
        console.error(traceback.format_exc())
        raise
