# webapp/sitecustomize.py

import asyncio
from js import console  # type: ignore

console.log("[sitecustomize] importing main()")

# async entrypoint
from main import main as _main  # resolves to /main.py in the APK

console.log("[sitecustomize] launching main()")
asyncio.run(_main())
