"""
Text-based interface for Player Game Mode
"""

from player_game import PlayerGame


def print_game_state(state: dict) -> None:
    """Print game state in readable format."""
    print("\n" + "=" * 80)
    print(f"EPISODE {state['episode']} | TURN {state['turn']}")
    print("=" * 80)
    
    # Player status
    player = state["player"]
    print(f"\n💰 YOUR PORTFOLIO")
    print(f"   Cash: ${player['cash']:.2f}")
    print(f"   Holdings: ${player['holdings_value']:.2f}")
    print(f"   Net Worth: ${player['net_worth']:.2f}")
    
    if player["holdings"]:
        print("   Holdings Breakdown:")
        for coin_id, units in player["holdings"].items():
            if units > 0:
                print(f"      {coin_id}: {units:.2f} units")
    
    # Leaderboard
    print(f"\n📊 LEADERBOARD")
    print(f"   {'Rank':<6}{'Agent':<10}{'Net Worth':<15}{'Cash':<12}{'Performance':<12}")
    print("   " + "-" * 55)
    for entry in state["leaderboard"]:
        marker = "👤" if entry["is_player"] else " "
        perf_str = f"{entry['recent_performance']:+.1f}%"
        agent_label = f"Agent {entry['agent_id']}"
        print(f"   {marker}{entry['rank']:<5}{agent_label:<10}${entry['net_worth']:<14.2f}${entry['cash']:<11.2f}{perf_str:<12}")
    
    # Market
    print(f"\n🪙 MARKET")
    print(f"   {'Coin':<8}{'Price':<10}{'Trend':<10}{'Volatility':<12}{'Liquidity':<12}{'Cap':<12}")
    print("   " + "-" * 64)
    for coin in state["market"]:
        trend_icon = {"rising": "📈", "falling": "📉", "stable": "➡️"}.get(coin["trend"], "")
        vol_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(coin["volatility"], "")
        print(f"   {coin['coin_id']:<8}${coin['price']:<9.2f}{trend_icon:<10}{vol_icon:<12}${coin['liquidity']:<11.0f}${coin['market_cap']:<11.0f}")
    
    # Recent events and agent actions
    if state["recent_events"]:
        print(f"\n📰 RECENT EVENTS & AGENT ACTIONS")
        for event in state["recent_events"][-5:]:
            print(f"   • {event}")
    
    if state["highlights"]:
        print(f"\n⚠️  HIGHLIGHTS")
        for item in state["highlights"]:
            print(f"   ⚡ {item}")
    
    # Available actions
    if state["waiting_for_player"]:
        print(f"\n🎮 YOUR ACTIONS")
        actions = state["available_actions"]
        
        # Group by type
        basic = [a for a in actions if not a.get("coin_id")]
        coin_actions = [a for a in actions if a.get("coin_id")]
        
        print("   Basic Actions:")
        for i, action in enumerate(basic[:5], 1):
            status = "✓" if action["enabled"] else "✗"
            print(f"   [{i}] {status} {action['label']} - {action['description']}")
        
        print("\n   Coin Actions (type 'c' to see):")
        print(f"      {len(coin_actions)} actions available for specific coins")
        
        print("\n   Enter action number or 'help' for more options")


def get_player_choice(state: dict, game: PlayerGame) -> bool:
    """Get player's action choice. Returns True if action submitted."""
    if not state["waiting_for_player"]:
        return False
    
    actions = state["available_actions"]
    basic = [a for a in actions if not a.get("coin_id")]
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == "help":
        print("\n📖 ACTION HELP")
        print("   Work - Earn steady income")
        print("   Mine - Generate new coins (costly but profitable)")
        print("   Hold - Wait and observe")
        print("   Buy/Sell - Trade coins")
        print("   Trend/Speculate - Advanced trading")
        print("   Create - Launch your own coin (expensive)")
        print("\n   Type 'm' to see mining options")
        print("   Type 'b' to see buy options")
        print("   Type 's' to see sell options")
        print("   Type 'c' to see all coin actions")
        return False
    
    if choice == "m":
        mining = [a for a in actions if a["type"] == "mine"]
        print("\n⛏️ MINING OPTIONS")
        for i, action in enumerate(mining, 1):
            print(f"   [{i}] {action['label']}")
        return False
    
    if choice == "b":
        buys = [a for a in actions if a["type"] == "buy"]
        print("\n💰 BUY OPTIONS")
        for i, action in enumerate(buys[:10], 1):
            print(f"   [{i}] {action['label']} - {action['description']}")
        return False
    
    if choice == "s":
        sells = [a for a in actions if a["type"] == "sell"]
        print("\n💸 SELL OPTIONS")
        for i, action in enumerate(sells[:10], 1):
            print(f"   [{i}] {action['label']} - {action['description']}")
        return False
    
    if choice == "c":
        coin_actions = [a for a in actions if a.get("coin_id")]
        print("\n🪙 COIN ACTIONS")
        for i, action in enumerate(coin_actions[:20], 1):
            status = "✓" if action["enabled"] else "✗"
            print(f"   [{i}] {status} {action['label']}")
        return False
    
    # Try to parse as number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(basic):
            action = basic[idx]
            if action["enabled"]:
                game.submit_player_action(
                    action["type"],
                    action.get("coin_id"),
                    action.get("fraction"),
                )
                print(f"\n✓ Action submitted: {action['label']}")
                return True
            else:
                print("✗ That action is not available right now")
        else:
            print("✗ Invalid choice")
    except ValueError:
        pass
    
    return False


def run_game() -> None:
    """Run the text-based player game."""
    print("=" * 80)
    print("🎮 COINGAME - PLAYER MODE")
    print("=" * 80)
    print("\nCompete against AI agents in a dynamic crypto economy!")
    print("Make money through trading, mining, and strategic investments.")
    print("\nControls:")
    print("   - Enter number to choose action")
    print("   - 'help' for action descriptions")
    print("   - 'm/b/s/c' for specific action categories")
    print("   - 'q' to quit")
    
    # Get configuration
    try:
        num_ai = int(input("\nNumber of AI opponents (5-120, default 119): ") or "119")
        num_ai = max(5, min(120, num_ai))
    except ValueError:
        num_ai = 119
    
    # Initialize game
    game = PlayerGame(num_ai_agents=num_ai)
    game.reset()
    
    print(f"\n✓ Game started with {num_ai} AI opponents")
    print("Good luck!\n")
    
    # Main game loop
    while not game.game_over:
        state = game.step()
        
        if state["game_over"]:
            print("\n🏁 GAME OVER")
            stats = game.get_player_statistics()
            print(f"   Episodes completed: {stats.get('episodes_completed', 0)}")
            print(f"   Total turns: {stats.get('total_turns', 0)}")
            print(f"   Win rate: {stats.get('win_rate', 0):.1f}%")
            break
        
        print_game_state(state)
        
        if state["waiting_for_player"]:
            # Get player input
            while not get_player_choice(state, game):
                pass
            
            # Process the turn
            game.process_full_turn()
            game.next_turn()
        else:
            # AI turn, auto-advance
            game.next_turn()
            input("\nPress Enter to continue...")
    
    print("\nThanks for playing!")


if __name__ == "__main__":
    run_game()
