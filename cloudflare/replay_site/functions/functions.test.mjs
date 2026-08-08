import assert from "node:assert/strict";
import test from "node:test";

import { onRequestGet as getReplay } from "./api/replay-data/[game_id].js";
import { onRequestGet as getCatalog } from "./api/replays.js";


function bucket(objects) {
  return {
    requested: [],
    async get(key) {
      this.requested.push(key);
      const body = objects[key];
      return body === undefined ? null : { body, httpEtag: `"${key}"` };
    },
  };
}


test("catalog function returns the R2 catalog", async () => {
  const replays = bucket({ "catalog.json": '[{"game_id":"game-a"}]' });
  const response = await getCatalog({ env: { REPLAYS: replays } });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
  assert.deepEqual(await response.json(), [{ game_id: "game-a" }]);
  assert.deepEqual(replays.requested, ["catalog.json"]);
});


test("replay function returns a compressed game bundle", async () => {
  const payload = new Uint8Array([31, 139, 8, 0]);
  const replays = bucket({ "games/game-a.json.gz": payload });
  const response = await getReplay({
    env: { REPLAYS: replays },
    params: { game_id: "game-a" },
  });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-encoding"), "gzip");
  assert.deepEqual(
    new Uint8Array(await response.arrayBuffer()),
    payload,
  );
  assert.deepEqual(replays.requested, ["games/game-a.json.gz"]);
});


test("replay function rejects unsafe keys before touching R2", async () => {
  const replays = bucket({});
  const response = await getReplay({
    env: { REPLAYS: replays },
    params: { game_id: "../secret" },
  });

  assert.equal(response.status, 400);
  assert.deepEqual(replays.requested, []);
});
