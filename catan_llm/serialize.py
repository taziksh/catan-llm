"""Renders an Observation as a model prompt.

Stateless: one self-contained prompt per decision. Options keep
legal_actions order — the index is the training label.
"""

from catan_llm.geometry import NODE_TILES
from catan_llm.observe import Observation, observe
from catan_llm.schema import ActionType, DecisionRecord, GameRecord, Resource

RESOURCES = [r.value for r in Resource]


def _word(resource):
    return resource.lower()


def _freqdeck_text(counts):
    return ", ".join(f"{n} {_word(r)}" for r, n in zip(RESOURCES, counts) if n)


def _hand_text(hand):
    return " ".join(f"{_word(r.value)}:{n}" for r, n in hand.items())


def _devs_text(devs):
    parts = [f"{card.value.lower()} x{n}" for card, n in devs.items() if n]
    return ", ".join(parts) or "none"


def _plural(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _tile_labels(obs: Observation):
    return {
        t.id: f"{_word(t.resource.value)}-{t.number}" if t.resource else "desert"
        for t in obs.layout.tiles
    }


def _own_nodes(obs: Observation):
    nodes = {n for n, color, _ in obs.board.buildings if color == obs.actor}
    for u, v, color in obs.board.roads:
        if color == obs.actor:
            nodes |= {u, v}
    return nodes


def option_text(action_type, payload, tiles, own_nodes):
    def node_text(node):
        return ", ".join(tiles[t] for t in NODE_TILES[node])

    match action_type:
        case ActionType.ROLL:
            return "roll the dice"
        case ActionType.END_TURN:
            return "end turn"
        case ActionType.BUY_DEVELOPMENT_CARD:
            return "buy a development card"
        case ActionType.PLAY_KNIGHT_CARD:
            return "play knight"
        case ActionType.PLAY_ROAD_BUILDING:
            return "play road building"
        case ActionType.CANCEL_TRADE:
            return "cancel trade"
        case ActionType.BUILD_SETTLEMENT:
            return f"build settlement at node {payload} (adjacent: {node_text(payload)})"
        case ActionType.BUILD_CITY:
            return f"upgrade to city at node {payload} (adjacent: {node_text(payload)})"
        case ActionType.BUILD_ROAD:
            new = [n for n in payload if n not in own_nodes] or list(payload)
            opens = "; ".join(f"opens node {n}: {node_text(n)}" for n in new)
            return f"build road {payload[0]}-{payload[1]} ({opens})"
        case ActionType.MOVE_ROBBER:
            tile, victim = payload
            steal = f", steal from {victim}" if victim else ""
            return f"move robber to tile {tile} ({tiles[tile]}){steal}"
        case ActionType.DISCARD_RESOURCE:
            return f"discard {_word(payload)}"
        case ActionType.PLAY_MONOPOLY:
            return f"play monopoly on {_word(payload)}"
        case ActionType.PLAY_YEAR_OF_PLENTY:
            return f"play year of plenty for {' + '.join(_word(r) for r in payload)}"
        case ActionType.MARITIME_TRADE:
            gives = [r for r in payload[:4] if r is not None]
            return f"trade {len(gives)} {_word(gives[0])} for 1 {_word(payload[4])}"
        case ActionType.OFFER_TRADE:
            return (
                f"offer trade: give {_freqdeck_text(payload[:5])} "
                f"for {_freqdeck_text(payload[5:10])}"
            )
        case ActionType.ACCEPT_TRADE:
            return "accept the trade offer"
        case ActionType.REJECT_TRADE:
            return "reject the trade offer"
        case ActionType.CONFIRM_TRADE:
            return f"confirm trade with {payload[10]}"
    raise ValueError(action_type)


def _board_lines(obs: Observation, tiles):
    tile_list = " ".join(f"{t}:{label}" for t, label in tiles.items())
    ports = ", ".join(
        f"{_word(p.resource.value) if p.resource else 'any'}@{p.nodes[0]}-{p.nodes[1]}"
        for p in obs.layout.ports
    )
    buildings = ", ".join(
        f"{color.value} {kind.value.lower()}@{node}"
        for node, color, kind in obs.board.buildings
    )
    roads_by_color = {}
    for u, v, color in obs.board.roads:
        roads_by_color.setdefault(color.value, []).append(f"{u}-{v}")
    roads = " | ".join(f"{c} {' '.join(edges)}" for c, edges in roads_by_color.items())
    return [
        "BOARD",
        f"tiles: {tile_list}",
        f"ports (resource=2:1, any=3:1): {ports}",
        f"robber: tile {obs.board.robber}",
        f"buildings: {buildings or '-'}",
        f"roads: {roads or '-'}",
    ]


def _badges(view):
    return (" | longest road" if view.has_longest_road else "") + (
        " | largest army" if view.has_largest_army else ""
    )


def _player_lines(obs: Observation):
    you = obs.you
    lines = [
        "PLAYERS",
        f"you ({obs.actor.value}): hand {_hand_text(you.hand)} | "
        f"dev cards: {_devs_text(you.devs_in_hand)} | "
        f"played: {_devs_text(you.devs_played)} | "
        f"{you.vps_actual} VP | road length {you.road_len}{_badges(you)}",
    ]
    for opp in obs.opponents:
        lines.append(
            f"{opp.color.value}: {_plural(opp.card_count, 'card')} | "
            f"{_plural(opp.dev_card_count, 'dev card')} | "
            f"played: {_devs_text(opp.devs_played)} | "
            f"{opp.vps_public} VP | road length {opp.road_len}{_badges(opp)}"
        )
    lines.append(
        f"bank: {_hand_text(obs.bank.resources)} | dev cards: {obs.bank.dev_cards_left}"
    )
    return lines


def observation_to_prompt(obs: Observation) -> str:
    tiles = _tile_labels(obs)
    own_nodes = _own_nodes(obs)
    intro = (
        f"You are {obs.actor.value} in a {1 + len(obs.opponents)}-player game of "
        f"Catan. First to {obs.vps_to_win} victory points wins."
    )
    if not obs.trading:
        intro += " No trading between players; maritime (bank/port) trades only."
    options = [
        f"{i}. {option_text(action_type, payload, tiles, own_nodes)}"
        for i, (action_type, payload) in enumerate(obs.legal_actions)
    ]
    lines = [
        intro,
        f"turn {obs.turn} | phase: {obs.phase.value}",
        "turn order: " + " -> ".join(
            f"{p.value} (you)" if p == obs.actor else p.value for p in obs.turn_order
        ),
        "",
        *_board_lines(obs, tiles),
        "",
        *_player_lines(obs),
        "",
        "YOUR OPTIONS",
        *options,
        "",
        'Reply with "answer: <option number>".',
    ]
    return "\n".join(lines)


def decision_to_prompt(game: GameRecord, decision: DecisionRecord) -> str:
    return observation_to_prompt(observe(game, decision))
