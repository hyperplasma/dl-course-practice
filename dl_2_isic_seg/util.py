import sys

def draw_progress_bar(cur, total, bar_len=50):
    """
        Print progress bar during training
    """
    cur_len = int(cur / total * bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()