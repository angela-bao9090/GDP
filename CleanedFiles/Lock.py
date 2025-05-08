import asyncio


class Lock:
    def __init__(self):
        self.passive = 0
        self.activeLock = asyncio.Lock()
        self.lock = asyncio.Lock()
        self.passivesDone = asyncio.Event()
        self.passivesDone.set()

    async def acquirePassiveLock(self):
        async with self.lock:
            self.passive += 1
            self.passivesDone.clear()

    async def releasePassiveLock(self):
        async with self.lock:
            self.passive -= 1
            if self.passive == 0:
                self.passivesDone.set()

    async def acquireActiveLock(self):
        await self.activeLock.acquire()
        await self.passivesDone.wait()

    def releaseActiveLock(self):
        self.activeLock.release()
