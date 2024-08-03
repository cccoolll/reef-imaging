import asyncio
from hypha_rpc import *

async def main():
    server = await connect_to_server({"server_url": "https://ai.imjoy.io"})

    # Get an existing service
    # Since "hello-world" is registered as a public service, we can access it using only the name "hello-world"
    svc = await server.get_service("built-in")
    rtc_server = await get_rtc_service(server, "squid-control-rtc")
    print(rtc_server)
    print("Connected to server, got service:", svc)
    print("Connected to server, got rtc service:", rtc_server)
if __name__ == "__main__":
    asyncio.run(main())