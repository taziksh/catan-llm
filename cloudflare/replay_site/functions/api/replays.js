export async function onRequestGet({ env }) {
  const object = await env.REPLAYS.get("catalog.json");
  if (object === null) {
    return Response.json({ error: "replay catalog not found" }, { status: 404 });
  }

  const headers = new Headers({
    "content-type": "application/json; charset=utf-8",
    "cache-control": "public, max-age=300",
  });
  if (object.httpEtag) headers.set("etag", object.httpEtag);
  return new Response(object.body, { headers });
}
