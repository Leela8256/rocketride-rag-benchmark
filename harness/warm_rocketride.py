"""One PDF through the RocketRide ingestion pipeline without the store, so that the engine installs the packages
of the nodes the benchmark uses and downloads the embedding model. Used once, while the images are built
(infra/build_images.sh); the container is then saved as the image every run starts from.

Runs in the client container. Usage: warm_rocketride.py <port> <pdf file>
"""
import asyncio
import json
import sys
from pathlib import Path


async def main(port, pdf):
    from rocketride import RocketRideClient
    client = RocketRideClient(uri='ws://127.0.0.1:%d/task/service' % port, auth='local-dev')
    await client.connect()
    token = None
    try:
        pipe = json.loads(Path('/pipelines/ingest_no_store.pipe').read_text())
        token = (await client.use(pipeline=pipe, threads=4))['token']
        body = Path(pdf).read_bytes()
        target = await client.pipe(token, objinfo={'name': Path(pdf).name, 'size': len(body)}, mime_type='application/pdf')
        await target.open()
        await target.write(body)
        result = await asyncio.wait_for(target.close(), 3600)
        text = json.dumps(result)
        assert 'page_content' in text and 'embedding' in text, 'the warm-up returned no chunks: ' + text[:400]
        print('WARM_OK', len(text), 'bytes returned')
    finally:
        if token:
            try:
                await asyncio.wait_for(client.terminate(token), 30)
            except Exception as e:
                print('TERMINATE', repr(e))
        await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[1]), sys.argv[2]))
