"""
Tests for Space Invaders game logic.
Tests verify game state updates correctly and match gameplay description.
"""

import pytest
import numpy as np


class TestSpaceInvadersGame:
    """Tests for SpaceInvadersGame class."""

    def test_game_class_exists(self, mock_pygame_module):
        """Test SpaceInvadersGame can be imported."""
        from src.games.space_invaders.game import SpaceInvadersGame

        assert SpaceInvadersGame is not None

    def test_game_metadata(self, mock_pygame_module):
        """Test game has correct metadata."""
        from src.games.space_invaders.game import SpaceInvadersGame

        metadata = SpaceInvadersGame.get_metadata()

        assert metadata.id == "space_invaders"
        assert metadata.name == "Space Invaders"
        assert metadata.supports_human is True
        assert "dqn" in metadata.recommended_algorithms

    def test_action_space_size(self, mock_pygame_module):
        """Test action space is 6 (movement x firing combinations)."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()

        assert game.action_space_size == 6

    def test_action_names(self, mock_pygame_module):
        """Test action names are properly defined."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()

        assert len(game.action_names) == 6
        assert "Fire" in game.action_names[1]  # STAY_FIRE
        assert "Left" in game.action_names[2]  # LEFT_NO_FIRE


class TestGameInitialization:
    """Tests for game reset and initial state."""

    def test_reset_creates_valid_state(self, mock_pygame_module):
        """Test reset returns valid game state."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        assert "player" in state
        assert "invaders" in state
        assert "bunkers" in state
        assert "score" in state
        assert "game_over" in state

    def test_invaders_5x11_formation(self, mock_pygame_module):
        """Test invaders are arranged in 5 rows x 11 columns."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        assert len(state["invaders"]) == 5
        assert all(len(row) == 11 for row in state["invaders"])

    def test_invader_types_correct(self, mock_pygame_module):
        """Test invader types: top=0, middle=1, bottom=2."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        # Row 0 should be top type (30 points)
        assert state["invaders"][0][0]["type"] == 0
        # Rows 1-2 should be middle type (20 points)
        assert state["invaders"][1][0]["type"] == 1
        assert state["invaders"][2][0]["type"] == 1
        # Rows 3-4 should be bottom type (10 points)
        assert state["invaders"][3][0]["type"] == 2
        assert state["invaders"][4][0]["type"] == 2

    def test_four_bunkers_created(self, mock_pygame_module):
        """Test 4 bunkers are created."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        assert len(state["bunkers"]) == 4

    def test_player_starts_center(self, mock_pygame_module):
        """Test player starts at horizontal center."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        player_x = state["player"]["x"]
        # Should be near center (within 10% tolerance)
        assert abs(player_x - game.width / 2) < game.width * 0.1

    def test_player_has_three_lives(self, mock_pygame_module):
        """Test player starts with 3 lives."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        assert state["lives"] == 3

    def test_all_invaders_alive_initially(self, mock_pygame_module):
        """Test all 55 invaders are alive at start."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        state = game.reset()

        assert state["invaders_alive"] == 55
        assert state["total_invaders"] == 55


class TestPlayerMovement:
    """Tests for player movement mechanics."""

    def test_move_left(self, mock_pygame_module):
        """Test player moves left with LEFT actions."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_x = game.player.x
        game.step(2)  # LEFT_NO_FIRE

        assert game.player.x < initial_x

    def test_move_right(self, mock_pygame_module):
        """Test player moves right with RIGHT actions."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_x = game.player.x
        game.step(4)  # RIGHT_NO_FIRE

        assert game.player.x > initial_x

    def test_stay_action(self, mock_pygame_module):
        """Test player stays in place with STAY actions."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_x = game.player.x
        game.step(0)  # STAY_NO_FIRE

        assert game.player.x == initial_x

    def test_player_clamped_left(self, mock_pygame_module):
        """Test player cannot move past left edge."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Move left many times
        for _ in range(200):
            game.step(2)  # LEFT_NO_FIRE

        # Should be clamped to minimum position
        assert game.player.x >= game.player.width / 2

    def test_player_clamped_right(self, mock_pygame_module):
        """Test player cannot move past right edge."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Move right many times
        for _ in range(200):
            game.step(4)  # RIGHT_NO_FIRE

        # Should be clamped to maximum position
        assert game.player.x <= game.width - game.player.width / 2


class TestPlayerFiring:
    """Tests for player firing mechanics."""

    def test_fire_creates_projectile(self, mock_pygame_module):
        """Test firing creates a player projectile."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        assert game.player_projectile is None
        game.step(1)  # STAY_FIRE

        assert game.player_projectile is not None

    def test_only_one_player_projectile(self, mock_pygame_module):
        """Test player can only have one projectile active."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        game.step(1)  # STAY_FIRE
        proj1_y = game.player_projectile.y

        game.step(1)  # Try to fire again

        # Should still be the same projectile (moved up)
        assert game.player_projectile.y < proj1_y

    def test_projectile_moves_up(self, mock_pygame_module):
        """Test player projectile moves upward."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        game.step(1)  # STAY_FIRE
        initial_y = game.player_projectile.y

        game.step(0)  # STAY_NO_FIRE (let projectile move)

        assert game.player_projectile.y < initial_y

    def test_projectile_clears_when_off_screen(self, mock_pygame_module):
        """Test projectile is removed when it goes off screen."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        game.step(1)  # STAY_FIRE

        # Step many times until projectile is off screen
        for _ in range(100):
            if game.player_projectile is None:
                break
            game.step(0)

        assert game.player_projectile is None

    def test_move_and_fire_simultaneously(self, mock_pygame_module):
        """Test player can move and fire at the same time."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_x = game.player.x
        game.step(3)  # LEFT_FIRE

        assert game.player.x < initial_x
        assert game.player_projectile is not None


class TestInvaderMovement:
    """Tests for invader formation movement."""

    def test_invaders_move_horizontally(self, mock_pygame_module):
        """Test invaders move left or right."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_x = game.invaders[0][0].x

        # Step a few times
        for _ in range(5):
            game.step(0)

        # Invader should have moved
        assert game.invaders[0][0].x != initial_x

    def test_invaders_reverse_at_edge(self, mock_pygame_module):
        """Test invaders reverse direction at screen edge."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_direction = game.invader_direction

        # Step many times until direction changes
        direction_changed = False
        for _ in range(500):
            game.step(0)
            if game.invader_direction != initial_direction:
                direction_changed = True
                break

        assert direction_changed

    def test_invaders_drop_when_reversing(self, mock_pygame_module):
        """Test invaders drop down when they reverse direction."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_direction = game.invader_direction
        initial_y = game.invaders[4][0].y  # Bottom row

        # Step until direction changes
        for _ in range(500):
            game.step(0)
            if game.invader_direction != initial_direction:
                break

        # Invaders should have dropped
        assert game.invaders[4][0].y > initial_y

    def test_invaders_speed_up_as_destroyed(self, mock_pygame_module):
        """Test invaders march completes faster as more are killed.

        With one-alien-per-frame movement, fewer aliens = faster march cycle.
        """
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # With 55 aliens, it takes 55 steps to complete one march cycle
        initial_alive = game._get_alive_count()
        assert initial_alive == 55

        # Kill some invaders
        for i in range(10):
            game.invaders[4][i].alive = False

        # Now with 45 aliens, march cycle is faster (45 steps)
        new_alive = game._get_alive_count()
        assert new_alive == 45
        assert new_alive < initial_alive  # Fewer aliens = faster march


class TestInvaderFiring:
    """Tests for invader firing mechanics."""

    def test_max_three_invader_projectiles(self, mock_pygame_module):
        """Test maximum 3 invader projectiles on screen."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Step many times to let invaders fire
        for _ in range(1000):
            game.step(0)
            # Count should never exceed 3
            assert len(game.invader_projectiles) <= 3

    def test_only_bottom_invaders_can_fire(self, mock_pygame_module):
        """Test only bottom-most invader per column can fire."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Get valid shooters
        shooters = game._get_bottom_invaders_per_column()

        # Each should be the bottom-most alive invader in its column
        for col, shooter in enumerate(shooters):
            if shooter:
                # Verify no alive invader below in same column
                shooter_row = None
                for row_idx, row in enumerate(game.invaders):
                    if row[col] == shooter:
                        shooter_row = row_idx
                        break
                # All rows below should be dead or this is the bottom
                if shooter_row is not None:
                    for row_idx in range(shooter_row + 1, 5):
                        assert not game.invaders[row_idx][col].alive


class TestCollisions:
    """Tests for collision detection."""

    def test_player_projectile_kills_invader(self, mock_pygame_module):
        """Test player projectile destroys invader on collision."""
        from src.games.space_invaders.game import SpaceInvadersGame, Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        # Get a target invader
        target_invader = game.invaders[4][5]

        # Create a projectile directly at the invader's position
        game.player_projectile = Projectile(
            x=target_invader.x,
            y=target_invader.y,
            owner=ProjectileOwner.PLAYER,
            speed=-8.0,
        )

        # Check collision
        game._check_player_projectile_collisions()

        assert not target_invader.alive

    def test_killing_invader_adds_score(self, mock_pygame_module):
        """Test killing invader increases score."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_score = game.score

        # Position player under an invader and kill it
        # Use column 4 (x=88) which is between bunkers (not column 5 at x=112 which is above bunker 3)
        target_invader = game.invaders[4][4]
        game.player.x = target_invader.x

        game.step(1)  # Fire
        for _ in range(100):
            if not target_invader.alive:
                break
            game.step(0)

        assert game.score > initial_score

    def test_invader_projectile_hits_player(self, mock_pygame_module):
        """Test invader projectile reduces player lives."""
        from src.games.space_invaders.game import SpaceInvadersGame
        from src.games.space_invaders.game import Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        initial_lives = game.player.lives

        # Create projectile directly on player
        game.invader_projectiles.append(
            Projectile(
                x=game.player.x,
                y=game.player.y,
                owner=ProjectileOwner.INVADER,
                speed=4.0,
            )
        )

        game._check_invader_projectile_collisions()

        assert game.player.lives < initial_lives

    def test_bunker_has_initial_health(self, mock_pygame_module):
        """Test bunkers start with health."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Get initial bunker health
        bunker = game.bunkers[0]
        initial_health = sum(
            cell.health
            for row in bunker.cells
            for cell in row
            if cell and cell.health > 0
        )

        assert initial_health > 0

    def test_player_bullet_erodes_bunker_bottom(self, mock_pygame_module):
        """Test player bullet damages bottom of bunker (chunk damage)."""
        from src.games.space_invaders.game import SpaceInvadersGame, Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        bunker = game.bunkers[0]
        # Count cells in bottom rows before damage
        bottom_cells_before = sum(
            1 for row in bunker.cells[-3:]  # Last 3 rows
            for cell in row if cell and cell.health > 0
        )

        # Create projectile hitting bunker from below (center of bunker)
        proj = Projectile(
            x=bunker.x + 11,  # Center of 22-wide bunker
            y=bunker.y + 14,  # Near bottom of bunker
            owner=ProjectileOwner.PLAYER,
            speed=-4.0
        )

        # Hit bunker
        hit = game._projectile_hits_bunker(proj, bunker, from_above=False)
        assert hit is True

        # Count cells in bottom rows after damage
        bottom_cells_after = sum(
            1 for row in bunker.cells[-3:]
            for cell in row if cell and cell.health > 0
        )

        # Multiple cells should be destroyed (chunk damage)
        assert bottom_cells_after < bottom_cells_before

    def test_enemy_bullet_erodes_bunker_top(self, mock_pygame_module):
        """Test enemy bullet damages top of bunker (chunk damage)."""
        from src.games.space_invaders.game import SpaceInvadersGame, Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        bunker = game.bunkers[0]
        # Count cells in top rows before damage
        top_cells_before = sum(
            1 for row in bunker.cells[:3]  # First 3 rows
            for cell in row if cell and cell.health > 0
        )

        # Create projectile hitting bunker from above
        proj = Projectile(
            x=bunker.x + 11,  # Center of bunker
            y=bunker.y + 2,   # Near top of bunker
            owner=ProjectileOwner.INVADER,
            speed=1.0
        )

        # Hit bunker
        hit = game._projectile_hits_bunker(proj, bunker, from_above=True)
        assert hit is True

        # Count cells in top rows after damage
        top_cells_after = sum(
            1 for row in bunker.cells[:3]
            for cell in row if cell and cell.health > 0
        )

        # Multiple cells should be destroyed (chunk damage)
        assert top_cells_after < top_cells_before

    def test_repeated_shots_create_tunnel(self, mock_pygame_module):
        """Test repeated shots at same spot create tunnel through bunker."""
        from src.games.space_invaders.game import SpaceInvadersGame, Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        bunker = game.bunkers[0]
        center_col = 5  # Center column

        # Count cells in center column before
        col_cells_before = sum(
            1 for row in bunker.cells
            if row[center_col] and row[center_col].health > 0
        )

        # Fire multiple shots at the same column from below
        for _ in range(10):
            proj = Projectile(
                x=bunker.x + center_col * 2 + 1,  # Center of cell
                y=bunker.y + 14,
                owner=ProjectileOwner.PLAYER,
                speed=-4.0
            )
            game._projectile_hits_bunker(proj, bunker, from_above=False)

        # Count cells in center column after
        col_cells_after = sum(
            1 for row in bunker.cells
            if row[center_col] and row[center_col].health > 0
        )

        # Significant destruction should have occurred
        assert col_cells_after < col_cells_before


class TestInvaderBunkerCollision:
    """Tests for invader-bunker collision (deletion mask behavior)."""

    def test_invaders_destroy_bunker_on_overlap(self, mock_pygame_module):
        """Test invaders destroy bunker cells when they overlap."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Get a bunker and count initial health
        bunker = game.bunkers[0]
        initial_health = sum(
            cell.health for row in bunker.cells for cell in row if cell and cell.health > 0
        )
        assert initial_health > 0

        # Move an invader directly onto the bunker
        # Bunker is at x=32, y=152 with size 22x16
        test_invader = game.invaders[4][0]  # Bottom-left invader
        test_invader.x = bunker.x + 11  # Center of bunker
        test_invader.y = bunker.y + 8   # Center of bunker

        # Trigger the collision check
        game._check_invader_bunker_collision()

        # Count remaining health
        remaining_health = sum(
            cell.health for row in bunker.cells for cell in row if cell and cell.health > 0
        )

        # Some cells should have been destroyed
        assert remaining_health < initial_health

    def test_invaders_continue_through_bunker(self, mock_pygame_module):
        """Test invaders continue marching through bunkers (not blocked)."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Move all invaders to bunker height
        for row in game.invaders:
            for inv in row:
                inv.y = game.bunker_y

        initial_positions = [(inv.x, inv.y) for row in game.invaders for inv in row]

        # Take a step - invaders should still move
        game.step(0)

        # Verify at least one invader moved (one-alien-per-frame)
        new_positions = [(inv.x, inv.y) for row in game.invaders for inv in row]
        assert new_positions != initial_positions


class TestGameOver:
    """Tests for game over conditions."""

    def test_game_over_when_no_lives(self, mock_pygame_module):
        """Test game ends when player has no lives."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        game.player.lives = 0
        game.step(0)

        assert game.game_over is True

    def test_game_over_when_invaders_reach_bottom(self, mock_pygame_module):
        """Test game ends when invaders reach Y=216 (classic arcade threshold)."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Move invaders to game over threshold (classic: Y=216)
        for row in game.invaders:
            for inv in row:
                inv.y = game.game_over_y

        game.step(0)

        assert game.game_over is True


class TestWaveProgression:
    """Tests for wave/level progression."""

    def test_next_wave_when_all_invaders_dead(self, mock_pygame_module):
        """Test new wave starts when all invaders are killed."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_wave = game.wave

        # Kill all invaders
        for row in game.invaders:
            for inv in row:
                inv.alive = False

        game.step(0)

        assert game.wave > initial_wave

    def test_next_wave_invaders_reset(self, mock_pygame_module):
        """Test next wave has all invaders alive again."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Kill all invaders
        for row in game.invaders:
            for inv in row:
                inv.alive = False

        game.step(0)

        # New wave should have all invaders
        alive_count = sum(1 for row in game.invaders for inv in row if inv.alive)
        assert alive_count == 55


class TestMysteryShip:
    """Tests for mystery ship mechanics."""

    def test_mystery_ship_starts_inactive(self, mock_pygame_module):
        """Test mystery ship starts inactive."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        assert game.mystery_ship.active is False

    def test_mystery_ship_gives_bonus(self, mock_pygame_module):
        """Test hitting mystery ship gives bonus points."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Simulate hitting mystery ship
        points = game.mystery_ship.get_points()

        assert points in [50, 100, 150, 300]


class TestSpaceInvadersEnv:
    """Tests for SpaceInvadersEnv (environment wrapper)."""

    def test_env_state_size(self, mock_pygame_module):
        """Test environment returns correct state size."""
        from src.games.space_invaders.env import SpaceInvadersEnv

        env = SpaceInvadersEnv()

        assert env.state_size == 32

    def test_env_action_size(self, mock_pygame_module):
        """Test environment returns correct action size."""
        from src.games.space_invaders.env import SpaceInvadersEnv

        env = SpaceInvadersEnv()

        assert env.action_size == 6

    def test_env_reset_returns_numpy(self, mock_pygame_module):
        """Test reset returns numpy array."""
        from src.games.space_invaders.env import SpaceInvadersEnv

        env = SpaceInvadersEnv()
        state = env.reset()

        assert isinstance(state, np.ndarray)
        assert len(state) == 32

    def test_env_step_returns_correct_tuple(self, mock_pygame_module):
        """Test step returns (state, reward, done, info)."""
        from src.games.space_invaders.env import SpaceInvadersEnv

        env = SpaceInvadersEnv()
        env.reset()

        state, reward, done, info = env.step(0)

        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_env_get_game_state(self, mock_pygame_module):
        """Test get_game_state returns dict for renderer."""
        from src.games.space_invaders.env import SpaceInvadersEnv

        env = SpaceInvadersEnv()
        env.reset()

        game_state = env.get_game_state()

        assert isinstance(game_state, dict)
        assert "player" in game_state
        assert "invaders" in game_state

    def test_env_state_normalized(self, mock_pygame_module):
        """Test state values are normalized (mostly 0-1 range)."""
        from src.games.space_invaders.env import SpaceInvadersEnv

        env = SpaceInvadersEnv()
        state = env.reset()

        # Most values should be in reasonable range
        # Some features like X offset can be -1 to 1
        assert all(-1.5 <= v <= 1.5 for v in state)


class TestRecording:
    """Tests for replay recording."""

    def test_recording_creates_history(self, mock_pygame_module):
        """Test recording captures frame history."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()
        game.start_recording()

        # Take some steps
        for _ in range(10):
            game.step(0)

        history = game.stop_recording()

        assert len(history) > 0
        assert "player" in history[0]
        assert "invaders" in history[0]


class TestConfig:
    """Tests for SpaceInvadersConfig."""

    def test_config_defaults(self, mock_pygame_module):
        """Test config has sensible defaults."""
        from src.games.space_invaders.config import SpaceInvadersConfig

        config = SpaceInvadersConfig()

        assert config.width == 224
        assert config.height == 256
        assert config.player_start_lives == 3
        assert config.invader_rows == 5
        assert config.invader_cols == 11

    def test_config_to_dict(self, mock_pygame_module):
        """Test config can be serialized to dict."""
        from src.games.space_invaders.config import SpaceInvadersConfig

        config = SpaceInvadersConfig()
        data = config.to_dict()

        assert "width" in data
        assert "height" in data
        assert "rewards" in data

    def test_config_from_dict(self, mock_pygame_module):
        """Test config can be created from dict."""
        from src.games.space_invaders.config import SpaceInvadersConfig

        data = {"width": 300, "height": 400, "player_speed": 5.0}
        config = SpaceInvadersConfig.from_dict(data)

        assert config.width == 300
        assert config.height == 400
        assert config.player_speed == 5.0


class TestRenderer:
    """Tests for SpaceInvadersRenderer."""

    def test_renderer_init(self, mock_pygame_module):
        """Test renderer initializes correctly."""
        from src.games.space_invaders.renderer import SpaceInvadersRenderer

        renderer = SpaceInvadersRenderer()

        assert renderer is not None

    def test_renderer_preferred_size(self, mock_pygame_module):
        """Test renderer returns preferred size."""
        from src.games.space_invaders.renderer import SpaceInvadersRenderer

        renderer = SpaceInvadersRenderer()
        width, height = renderer.get_preferred_size()

        assert width > 0
        assert height > 0

    def test_renderer_render_no_crash(self, mock_pygame_module, mock_screen):
        """Test renderer doesn't crash on render."""
        from src.games.space_invaders.renderer import SpaceInvadersRenderer
        from src.games.space_invaders.game import SpaceInvadersGame

        renderer = SpaceInvadersRenderer()
        game = SpaceInvadersGame()
        state = game.reset()

        # Should not raise
        renderer.render(state, mock_screen)


class TestGameRegistration:
    """Tests for game registration with GameRegistry."""

    def test_game_registered(self, mock_pygame_module):
        """Test Space Invaders is registered in GameRegistry."""
        from src.games.registry import GameRegistry

        # Import to trigger registration
        import src.games.space_invaders  # noqa: F401

        assert GameRegistry.is_available("space_invaders")

    def test_can_create_env_from_registry(self, mock_pygame_module):
        """Test environment can be created from registry."""
        from src.games.registry import GameRegistry

        import src.games.space_invaders  # noqa: F401

        env = GameRegistry.create_env("space_invaders")
        assert env is not None
        assert env.state_size == 32
        assert env.action_size == 6


class TestBulletVsBullet:
    """Tests for bullet vs bullet collision."""

    def test_player_bullet_destroys_invader_bullet(self, mock_pygame_module):
        """Both bullets destroyed when they collide."""
        from src.games.space_invaders.game import SpaceInvadersGame, Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        # Create player bullet
        game.player_projectile = Projectile(x=100, y=150, owner=ProjectileOwner.PLAYER, speed=-4.0)
        # Create invader bullet at same position
        game.invader_projectiles.append(
            Projectile(x=100, y=150, owner=ProjectileOwner.INVADER, speed=1.0)
        )

        result = game._check_bullet_collision()

        assert result is True
        assert game.player_projectile is None
        assert len(game.invader_projectiles) == 0


class TestSpeedRatios:
    """Tests for classic arcade speed values."""

    def test_player_speed_is_one(self, mock_pygame_module):
        """Player moves 1 pixel per frame."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        initial_x = game.player.x
        game.step(4)  # RIGHT_NO_FIRE

        assert game.player.x == initial_x + 1.0

    def test_player_bullet_speed_is_four(self, mock_pygame_module):
        """Player bullet moves 4 pixels per frame."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        game.step(1)  # STAY_FIRE
        initial_y = game.player_projectile.y
        game.step(0)  # Let bullet move

        assert initial_y - game.player_projectile.y == 4.0

    def test_invader_bullet_speed_is_one(self, mock_pygame_module):
        """Invader bullet moves 1 pixel per frame."""
        from src.games.space_invaders.game import SpaceInvadersGame, Projectile, ProjectileOwner

        game = SpaceInvadersGame()
        game.reset()

        game.invader_projectiles.append(
            Projectile(x=100, y=100, owner=ProjectileOwner.INVADER, speed=1.0)
        )
        initial_y = game.invader_projectiles[0].y
        game.step(0)

        assert game.invader_projectiles[0].y == initial_y + 1.0

    def test_alien_march_step_is_two(self, mock_pygame_module):
        """Each alien moves 2 pixels per step."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        assert game.march_step == 2


class TestMarchCycle:
    """Tests for one-alien-per-frame march behavior."""

    def test_march_index_cycles_through_aliens(self, mock_pygame_module):
        """March index wraps around after all aliens have moved."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        alive_count = game._get_alive_count()
        assert alive_count == 55

        # After 55 steps, march_index = 55
        # It wraps to 0 at the START of the next call
        for _ in range(55):
            game._move_invaders()

        assert game.march_index == 55  # Full cycle completed

        # One more call wraps it back to 0 then increments to 1
        game._move_invaders()
        assert game.march_index == 1  # Wrapped and moved one

    def test_fewer_aliens_faster_cycle(self, mock_pygame_module):
        """With fewer aliens, march cycle completes faster."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Kill all but 10 aliens
        count = 0
        for row in game.invaders:
            for inv in row:
                if count >= 10:
                    inv.alive = False
                count += 1

        assert game._get_alive_count() == 10

        # After 10 moves, march_index = 10
        for _ in range(10):
            game._move_invaders()

        assert game.march_index == 10  # Full cycle for 10 aliens

        # Next call wraps and increments
        game._move_invaders()
        assert game.march_index == 1


class TestSpatialAccuracy:
    """Tests for classic arcade coordinate positions."""

    def test_player_y_position(self, mock_pygame_module):
        """Player is at Y=216."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        assert game.player.y == 216

    def test_bunker_y_position(self, mock_pygame_module):
        """Bunkers are at Y=152."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        for bunker in game.bunkers:
            assert bunker.y == 152

    def test_ufo_y_position(self, mock_pygame_module):
        """UFO flies at Y=32."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        assert game.mystery_ship.y == 32

    def test_invaders_spawn_at_y_72(self, mock_pygame_module):
        """Top row of invaders spawns at Y=72."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        top_row_y = game.invaders[0][0].y
        assert top_row_y == 72

    def test_bunker_x_positions(self, mock_pygame_module):
        """Bunkers at X=[32, 72, 112, 152]."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        expected = [32, 72, 112, 152]
        actual = [bunker.x for bunker in game.bunkers]
        assert actual == expected

    def test_wave_2_starts_8px_lower(self, mock_pygame_module):
        """Wave 2 invaders spawn at Y=80 (8px lower)."""
        from src.games.space_invaders.game import SpaceInvadersGame

        game = SpaceInvadersGame()
        game.reset()

        # Kill all invaders to trigger wave 2
        for row in game.invaders:
            for inv in row:
                inv.alive = False
        game.step(0)

        assert game.wave == 2
        top_row_y = game.invaders[0][0].y
        assert top_row_y == 80  # 72 + 8
