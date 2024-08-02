import asyncio
from hypha_rpc import connect_to_server

async def main():
    server = await connect_to_server({"server_url": "http://reef.aicell.io:9527"})

    # Get an existing service
    # Since "hello-world" is registered as a public service, we can access it using only the name "hello-world"
    svc = await server.get_service("microscope-control-squid-2")
    #ret = await svc.hello("John")
    #print(ret)
    print(f"Connected to the server,{svc}")
if __name__ == "__main__":
    asyncio.run(main())