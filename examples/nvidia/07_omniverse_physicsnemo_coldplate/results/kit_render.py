# Run inside NVIDIA Omniverse Kit / USD Composer (needs an RTX GPU). Written by pinneapple_tools.nvidia.
import asyncio
import os

import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

HERE = os.path.dirname(os.path.abspath(__file__))


async def main():
    ctx = omni.usd.get_context()
    await ctx.open_stage_async(os.path.join(HERE, 'coldplate.usda'))
    for _ in range(60):                       # let RTX converge
        await omni.kit.app.get_app().next_update_async()
    capture_viewport_to_file(get_active_viewport(), os.path.join(HERE, 'coldplate_rtx.png'))
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

asyncio.ensure_future(main())
