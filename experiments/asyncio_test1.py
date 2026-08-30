import time
import asyncio

# def task(name, seconds):
#     print("Starting", name)
#     time.sleep(seconds)
#     print("Finished", name)

async def task(name, seconds):
    print("Starting", name)
    await asyncio.sleep(seconds)
    print("Finished", name)


# def main():
#     task("A", 2)
#     task("B", 2)
#     task("C", 2)

async def main():
    await asyncio.gather(
        task("A", 2),
        task("B", 2),
        task("C", 2)
    )
    
if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print("Total time:", end - start)

