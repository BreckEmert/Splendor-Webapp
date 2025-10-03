# webapp/main.py
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

    # Agents + game (real inference model)
    model_path = _resolve_model_path()
    log(f"[boot] loading model: {model_path}")
    rl_agent = InferenceModel(model_path)
    human = HumanAgent()
    players = [("DDQN", rl_agent, 0), ("Human", human, 1)]

    game = GUIGame(players, rl_agent)
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
