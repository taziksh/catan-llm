# JSettlers2 attribution and license notice

The board-rendering portions of `replay.html` are adapted from
[`SOCBoardPanel.java`](https://github.com/jdmonin/JSettlers2/blob/main/src/main/java/soc/client/SOCBoardPanel.java)
in [JSettlers2](https://github.com/jdmonin/JSettlers2), commit
`12baebade8e206367b4b770de6d5a8b40fc5d726`.

Adapted elements include the hex border palette, dice-probability number tokens,
port windows, player palette, and the road, settlement, city, and
robber geometry. The rendered pastel `*Hex.gif` tiles and
`assets/jsettlers2/miscPort.gif` are copied from JSettlers2's pastel graphics
set. The viewer omits JSettlers2's port arrowheads.

JSettlers2 copyright:

- Copyright (C) 2003 Robert S. Thomas
- Portions Copyright (C) 2007–2026 Jeremy D Monin
- Portions Copyright (C) 2012–2013 Paul Bilnoski
- Portions Copyright (C) 2017 Ruud Poutsma

JSettlers2 is free software licensed under the GNU General Public License,
version 3 or (at your option) any later version. The adapted renderer in this
directory is distributed under the same terms. See [`COPYING-GPLv3.txt`](COPYING-GPLv3.txt)
for the complete license.

The underlying pastel terrain art is by qubodup and is also available under
CC BY-SA 3.0. See [`assets/pastel/LICENSE.md`](assets/pastel/LICENSE.md).
