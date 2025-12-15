"""
Tests for UI components (Button, Dropdown, Toggle).

These tests verify that the core UI building blocks
can be initialized and manipulated without crashing.
"""

import pytest


class TestButton:
    """Tests for the Button class."""

    def test_button_init(self, mock_pygame_module):
        """Test button initialization."""
        from src.visualization.ui_components import Button

        btn = Button(100, 200, 150, 50, "Test Button")

        assert btn.text == "Test Button"
        assert btn.rect.x == 100
        assert btn.rect.y == 200
        assert btn.rect.width == 150
        assert btn.rect.height == 50
        assert btn.hovered is False
        assert btn.callback is None

    def test_button_with_callback(self, mock_pygame_module):
        """Test button with callback function."""
        from src.visualization.ui_components import Button

        callback_called = []

        def my_callback():
            callback_called.append(True)

        btn = Button(0, 0, 100, 50, "Click Me", callback=my_callback)

        assert btn.callback is not None

    def test_button_set_position(self, mock_pygame_module):
        """Test button position update."""
        from src.visualization.ui_components import Button

        btn = Button(0, 0, 100, 50, "Test")
        btn.set_position(300, 400)

        assert btn.rect.x == 300
        assert btn.rect.y == 400

    def test_button_set_size(self, mock_pygame_module):
        """Test button size update."""
        from src.visualization.ui_components import Button

        btn = Button(0, 0, 100, 50, "Test")
        btn.set_size(200, 80)

        assert btn.rect.width == 200
        assert btn.rect.height == 80

    def test_button_font_property(self, mock_pygame_module):
        """Test that font property creates font on first access."""
        from src.visualization.ui_components import Button

        btn = Button(0, 0, 100, 50, "Test", font_size=32)

        # First access should create font
        font1 = btn.font
        assert font1 is not None

        # Second access should return same font
        font2 = btn.font
        assert font1 is font2


class TestDropdown:
    """Tests for the Dropdown class."""

    def test_dropdown_init(self, mock_pygame_module):
        """Test dropdown initialization."""
        from src.visualization.ui_components import Dropdown

        items = ["Option 1", "Option 2", "Option 3"]
        dropdown = Dropdown(100, 200, 250, items, label="Select:")

        assert dropdown.x == 100
        assert dropdown.y == 200
        assert dropdown.width == 250
        assert dropdown.label == "Select:"
        assert dropdown.items == items
        assert dropdown.selected_index == 0
        assert dropdown.expanded is False

    def test_dropdown_empty_items(self, mock_pygame_module):
        """Test dropdown with empty items list."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, [])

        assert dropdown.items == ["(none)"]
        assert dropdown.selected_index == 0

    def test_dropdown_set_position(self, mock_pygame_module):
        """Test dropdown position update."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, ["A", "B"])
        dropdown.set_position(150, 250)

        assert dropdown.x == 150
        assert dropdown.y == 250

    def test_dropdown_set_width(self, mock_pygame_module):
        """Test dropdown width update."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, ["A", "B"])
        dropdown.set_width(350)

        assert dropdown.width == 350

    def test_dropdown_get_selected(self, mock_pygame_module):
        """Test getting selected item."""
        from src.visualization.ui_components import Dropdown

        items = ["First", "Second", "Third"]
        dropdown = Dropdown(0, 0, 200, items)

        assert dropdown.get_selected() == "First"

        dropdown.selected_index = 2
        assert dropdown.get_selected() == "Third"

    def test_dropdown_get_selected_index(self, mock_pygame_module):
        """Test getting selected index."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, ["A", "B", "C"])

        assert dropdown.get_selected_index() == 0

        dropdown.selected_index = 1
        assert dropdown.get_selected_index() == 1

    def test_dropdown_refresh_items(self, mock_pygame_module):
        """Test refreshing items list."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, ["Old 1", "Old 2"])
        dropdown.selected_index = 1
        dropdown.expanded = True
        dropdown.scroll_offset = 1

        new_items = ["New A", "New B", "New C"]
        dropdown.refresh_items(new_items)

        assert dropdown.items == new_items
        assert dropdown.selected_index == 0  # Reset
        assert dropdown.expanded is False  # Closed
        assert dropdown.scroll_offset == 0  # Reset

    def test_dropdown_refresh_with_empty(self, mock_pygame_module):
        """Test refreshing with empty list."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, ["A", "B"])
        dropdown.refresh_items([])

        assert dropdown.items == ["(none)"]

    def test_dropdown_header_rect(self, mock_pygame_module):
        """Test header rect property."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(50, 100, 200, ["A"])

        rect = dropdown.header_rect
        assert rect.x == 50
        assert rect.y == 100
        assert rect.width == 200

    def test_dropdown_font_properties(self, mock_pygame_module):
        """Test that font properties create fonts on first access."""
        from src.visualization.ui_components import Dropdown

        dropdown = Dropdown(0, 0, 200, ["A"], font_size=24)

        # Test font
        font1 = dropdown.font
        font2 = dropdown.font
        assert font1 is font2

        # Test label_font
        lfont1 = dropdown.label_font
        lfont2 = dropdown.label_font
        assert lfont1 is lfont2


class TestToggle:
    """Tests for the Toggle class."""

    def test_toggle_init_default(self, mock_pygame_module):
        """Test toggle initialization with default state."""
        from src.visualization.ui_components import Toggle

        toggle = Toggle(100, 200, "Enable Feature")

        assert toggle.x == 100
        assert toggle.y == 200
        assert toggle.label == "Enable Feature"
        assert toggle.state is False
        assert toggle.is_on() is False

    def test_toggle_init_on(self, mock_pygame_module):
        """Test toggle initialization with on state."""
        from src.visualization.ui_components import Toggle

        toggle = Toggle(0, 0, "Setting", initial_state=True)

        assert toggle.state is True
        assert toggle.is_on() is True

    def test_toggle_set_position(self, mock_pygame_module):
        """Test toggle position update."""
        from src.visualization.ui_components import Toggle

        toggle = Toggle(0, 0, "Test")
        toggle.set_position(200, 300)

        assert toggle.x == 200
        assert toggle.y == 300

    def test_toggle_set_state(self, mock_pygame_module):
        """Test toggle state update."""
        from src.visualization.ui_components import Toggle

        toggle = Toggle(0, 0, "Test", initial_state=False)

        toggle.set_state(True)
        assert toggle.is_on() is True

        toggle.set_state(False)
        assert toggle.is_on() is False

    def test_toggle_rect_properties(self, mock_pygame_module):
        """Test toggle rect properties."""
        from src.visualization.ui_components import Toggle

        toggle = Toggle(50, 100, "Label")

        rect = toggle.toggle_rect
        assert rect.x == 50
        assert rect.y == 100
        assert rect.width == toggle.toggle_width
        assert rect.height == toggle.toggle_height

    def test_toggle_font_property(self, mock_pygame_module):
        """Test that font property creates font on first access."""
        from src.visualization.ui_components import Toggle

        toggle = Toggle(0, 0, "Test", font_size=24)

        font1 = toggle.font
        font2 = toggle.font
        assert font1 is font2


class TestThemeColors:
    """Tests for theme color constants."""

    def test_colors_are_rgb_tuples(self, mock_pygame_module):
        """Test that all color constants are valid RGB tuples."""
        from src.visualization.ui_components import (
            BG_COLOR, PANEL_COLOR, TEXT_COLOR, ACCENT_COLOR,
            ACCENT_ORANGE, WARNING_COLOR, DANGER_COLOR, SUCCESS_COLOR,
            HOVER_COLOR, BUTTON_COLOR, BUTTON_HOVER, BORDER_COLOR, DISABLED_COLOR
        )

        colors = [
            BG_COLOR, PANEL_COLOR, TEXT_COLOR, ACCENT_COLOR,
            ACCENT_ORANGE, WARNING_COLOR, DANGER_COLOR, SUCCESS_COLOR,
            HOVER_COLOR, BUTTON_COLOR, BUTTON_HOVER, BORDER_COLOR, DISABLED_COLOR
        ]

        for color in colors:
            assert isinstance(color, tuple), f"Color {color} is not a tuple"
            assert len(color) == 3, f"Color {color} doesn't have 3 components"
            for component in color:
                assert 0 <= component <= 255, f"Color component {component} out of range"
