import socket

import numpy as np

import game


class TcpClient:
    def __init__(self, mcts, sock=None, name="vamperouge"):
        self.mcts = mcts
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.name = name
        self.game = None
        self.race_coord = None
        self.is_werewolf = False
        self.player = None

    def connect(self, host, port):
        print(f"connecting to {host}:{port}")
        self.sock.connect((host, port))

    def send_name(self):
        msg = (
            "NME".encode("ascii")
            + len(self.name).to_bytes(1, byteorder="big")
            + self.name.encode("ascii")
        )
        print(f"sending {msg}")
        sent = self.sock.send(msg)
        if not sent:
            raise RuntimeError("socket connection broken")

    def send_move(self, move):
        print(f"sending move {move}")
        msg = "MOV".encode("ascii")
        msg += (1).to_bytes(1, byteorder="big")
        msg += (move.start.x).to_bytes(1, byteorder="big")
        msg += (move.start.y).to_bytes(1, byteorder="big")
        msg += (move.n).to_bytes(1, byteorder="big")
        msg += (move.end.x).to_bytes(1, byteorder="big")
        msg += (move.end.y).to_bytes(1, byteorder="big")
        print(f"sending {msg}")
        sent = self.sock.send(msg)
        if not sent:
            raise RuntimeError("socket connection broken")

    def _chunks_to_msg(self):
        chunks = []
        bytes_recd = 0
        # read at max 3 consecutive bytes
        while bytes_recd < 3:
            chunk = self.sock.recv(min(3 - bytes_recd, 2048))
            if chunk == b"":
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b"".join(chunks)

    def receive_msg(self):
        msg = self._chunks_to_msg()
        command = msg[:3].decode("ascii")
        print(f"received command {command}")
        if command == "SET":
            buf = self.sock.recv(2)
            n = int(buf[0])
            m = int(buf[1])
            print(f"set the map size to ({n}, {m})")
            self.game = game.State(height=n, width=m,)
        elif command == "HUM":
            buf = self.sock.recv(1)
            n = int(buf[0])
            coords = []
            for _ in range(n):
                buf = self.sock.recv(2)
                coords.append(game.Coordinates(int(buf[0]), int(buf[1])))
            self.game.hum(coords)
        elif command == "HME":
            buf = self.sock.recv(2)
            x = int(buf[0])
            y = int(buf[1])
            print(f"received race coordinates: ({x}, {y})")
            self.race_coord = game.Coordinates(x, y)
        elif command == "UPD":
            buf = self.sock.recv(1)
            n = int(buf[0])
            changes = []
            for _ in range(n):
                buf = self.sock.recv(5)
                x = int(buf[0])
                y = int(buf[1])
                humans = int(buf[2])
                vampires = int(buf[3])
                werewolves = int(buf[4])
                coord = game.Coordinates(x, y)
                changes.append((coord, humans, vampires, werewolves))
            self.game.upd(changes)
        elif command == "MAP":
            buf = self.sock.recv(1)
            n = int(buf[0])
            changes = []
            for _ in range(n):
                buf = self.sock.recv(5)
                x = int(buf[0])
                y = int(buf[1])
                humans = int(buf[2])
                vampires = int(buf[3])
                werewolves = int(buf[4])
                coord = game.Coordinates(x, y)
                if coord == self.race_coord and werewolves != 0:
                    self.is_werewolf = True
                    print("we are werewolves")
                changes.append((coord, humans, vampires, werewolves))
            self.game.upd(changes)
        elif command == "END":
            print("end of game")
            self.game = None
            self.race_coord = None
            self.is_werewolf = False
            self.player = None
        elif command == "BYE":
            print("server said bye")
        else:
            print(f"received unknown command {command}")
        return command

    def receive_specific_command(self, command):
        received = self.receive_msg()
        if received != command:
            raise RuntimeError(
                f"received unexpected command {received} (expected {command})"
            )

    def play(self):
        print("start playing")
        while True:
            received = self.receive_msg()
            if received == "UPD":
                move = self.player(self.game)
                self.send_move(move)
            elif received == "BYE":
                print("stopping the client")
                return
            elif received == "END":
                print("getting ready for next game")
                self.start()
            else:
                raise RuntimeError(f"received unexpected command {received}")

    def init_game(self):
        print("initializing game")
        self.send_name()
        self.receive_specific_command("SET")
        self.receive_specific_command("HUM")
        self.receive_specific_command("HME")
        self.receive_specific_command("MAP")

    def start(self):
        print("starting client")
        self.init_game()

        def player(state):
            race = -1 if self.is_werewolf else 1
            canon_state = game.get_canonical_form(state, race)
            action = np.argmax(self.mcts.get_move_probabilities(canon_state, temp=0))
            return state.action_to_move(action)

        self.player = player
        self.play()
