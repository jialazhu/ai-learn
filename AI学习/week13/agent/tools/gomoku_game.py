
class GomokuGame:
    """五子棋游戏类"""

    def __init__(self):
        """初始化游戏"""
        self.board_size = 15
        self.board = [[' ' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = '⚫'  # 黑棋先手
        self.game_over = False
        self.winner = None

    def display_board(self):
        """显示棋盘"""
        print('   ' + ' '.join([f'{i:2}' for i in range(self.board_size)]))
        print('  ' + '+' + '-' * (self.board_size * 3 - 1) + '+')

        for i in range(self.board_size):
            row_str = f'{i:2}|'
            for j in range(self.board_size):
                row_str += f' {self.board[i][j]} '
            row_str += '|'
            print(row_str)

        print('  ' + '+' + '-' * (self.board_size * 3 - 1) + '+')

    def is_valid_move(self, row, col):
        """检查移动是否有效"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row][col] == ' '

    def make_move(self, row, col):
        """下棋"""
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player

        # 检查是否获胜
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        # 检查是否平局
        elif self.is_board_full():
            self.game_over = True
            self.winner = None
        else:
            # 切换玩家
            self.current_player = '⚪' if self.current_player == '⚫' else '⚫'

        return True

    def check_win(self, row, col):
        """检查是否获胜（检查四个方向）"""
        player = self.board[row][col]

        # 检查方向：水平、垂直、对角线1、对角线2
        directions = [
            [(0, 1), (0, -1)],   # 水平
            [(1, 0), (-1, 0)],   # 垂直
            [(1, 1), (-1, -1)],  # 对角线1
            [(1, -1), (-1, 1)]   # 对角线2
        ]

        for direction in directions:
            count = 1  # 包括当前位置

            # 检查两个方向
            for dx, dy in direction:
                i, j = row, col
                while True:
                    i += dx
                    j += dy
                    if (0 <= i < self.board_size and
                        0 <= j < self.board_size and
                        self.board[i][j] == player):
                        count += 1
                    else:
                        break

            if count >= 5:
                return True

        return False

    def is_board_full(self):
        """检查棋盘是否已满"""
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def get_game_status(self):
        """获取游戏状态"""
        if not self.game_over:
            return f"当前玩家: {self.current_player}"
        elif self.winner:
            return f"游戏结束！{self.winner} 获胜！"
        else:
            return "游戏结束！平局！"

    def reset_game(self):
        """重置游戏"""
        self.board = [[' ' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = '⚫'
        self.game_over = False
        self.winner = None


def main():
    """游戏主函数"""
    game = GomokuGame()

    print("欢迎来到五子棋游戏！")
    print("⚫ 黑棋先手，⚪ 白棋后手")
    print("输入坐标格式：行 列 (例如: 7 7)")
    print("输入 'quit' 退出游戏，输入 'reset' 重新开始")
    print("-" * 50)

    while True:
        game.display_board()
        print(game.get_game_status())

        if game.game_over:
            print("是否重新开始？(y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                game.reset_game()
                continue
            elif choice == 'n':
                break
            else:
                continue

        try:
            user_input = input("请输入落子位置 (行 列): ").strip()

            if user_input.lower() == 'quit':
                print("游戏结束！")
                break
            elif user_input.lower() == 'reset':
                game.reset_game()
                continue

            row, col = map(int, user_input.split())

            if game.make_move(row, col):
                print(f"✓ {game.current_player if game.current_player == '⚪' else '⚫'} 在 ({row}, {col}) 落子成功")
            else:
                print("✗ 无效的落子位置，请重新输入")

        except ValueError:
            print("✗ 输入格式错误，请输入 '行 列' 格式")
        except Exception as e:
            print(f"✗ 发生错误: {e}")

        print("-" * 50)


if __name__ == "__main__":
    main()