const SAFE_GAME_ID = /^[A-Za-z0-9_.-]+$/;

export async function onRequestGet({ env, params }) {
  const gameId = String(params.game_id);
  if (!SAFE_GAME_ID.test(gameId)) {
    return Response.json({ error: "invalid game id" }, { status: 400 });
  }

  const object = await env.REPLAYS.get(`games/${gameId}.json.gz`);
  if (object === null) {
    return Response.json({ error: "replay not found" }, { status: 404 });
  }

  const headers = new Headers({
    "content-type": "application/json; charset=utf-8",
    "content-encoding": "gzip",
    "cache-control": "public, max-age=31536000, immutable",
  });
  if (object.httpEtag) headers.set("etag", object.httpEtag);
  return new Response(object.body, { headers, encodeBody: "manual" });
}
