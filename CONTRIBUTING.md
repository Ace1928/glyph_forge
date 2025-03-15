# Contributing to Glyph Forge ⚡

> *"Where precision meets purpose and structure enhances function."*

Welcome to the Glyph Forge contributor ecosystem. Your participation represents a commitment to structural integrity, algorithmic elegance, and transformative code. This document will guide you through our contribution process with exact specifications.

## 🔄 Eidosian Development Principles

Glyph Forge follows these core principles:

1. **Contextual Integrity** - Every element must serve a precise purpose
2. **Structure as Control** - Clean architectural boundaries with defined interfaces
3. **Exhaustive But Concise** - Complete functionality with minimal expression
4. **Humor as Cognitive Leverage** - Wit that enhances understanding
5. **Self-Awareness as Foundation** - Continuous improvement through reflection

## ⚙️ Development Environment Setup

```bash
# Clone repository
git clone https://github.com/Ace1928/glyph_forge.git
cd glyph_forge

# Create isolated environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
pytest
```

## 🧩 Contribution Workflow

### 1. Issue Alignment

Before implementation, establish context:

- **Issue first** - Create or reference an existing issue
- **Design alignment** - Match implementation to architectural patterns
- **Scope definition** - Clearly articulate boundaries and deliverables

### 2. Branch Naming System

```bash
# Sync with main branch
git checkout main
git pull origin main

# Create semantic branch
git checkout -b feature/specific-capability-name
```

Branch prefixes with purpose:

- `feature/` - New capabilities
- `fix/` - Error resolution
- `refactor/` - Structure improvement
- `perf/` - Performance enhancement
- `docs/` - Documentation update

### 3. Implementation Process

```bash
# Test-driven approach
# 1. Write tests
# 2. Implement functionality
# 3. Refactor for clarity

# Verification sequence
pytest

# Style enforcement
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### 4. Commit Structure

Commits must be atomic and descriptive:

```bash
# Format: <type>: <specific change description>
git commit -m "feat: implement adaptive character density mapping"
git commit -m "fix: resolve edge case in brightness calculation"
git commit -m "docs: clarify parameter constraints with examples"
```

Categories:

- **feat**: New functionality
- **fix**: Bug resolution
- **refactor**: Code restructuring
- **perf**: Performance improvement
- **docs**: Documentation enhancement
- **test**: Test coverage expansion
- **chore**: Maintenance tasks

### 5. Pull Request Process

```bash
# Push changes
git push -u origin feature/specific-capability-name

# Create PR via GitHub
# Use the template for consistency
```

## 🔍 Code Style & Structure

Glyph Forge maintains these standards:

### Python Style

- **Formatting**: Black (88 char line limit)
- **Imports**: isort with sections
- **Typing**: Comprehensive type annotations
- **Documentation**: Google-style docstrings

### Code Architecture

1. **Single-Purpose Functions** - Each function performs exactly one operation
2. **Elimination of Redundancy** - No duplicated logic
3. **Input Validation** - All parameters validated at interfaces
4. **Precise Exception Handling** - Specific, informative errors
5. **Performance Awareness** - Efficiency in critical paths

### Implementation Example

```python
def map_luminance_to_glyph(
    values: np.ndarray,
    charset: str,
    invert: bool = False,
    contrast: float = 1.0
) -> str:
    """
    Map luminance values to glyph characters with controlled density.
    
    Args:
        values: Array of luminance values (0-255)
        charset: Characters from darkest to lightest
        invert: Reverse the mapping direction
        contrast: Contrast adjustment factor
        
    Returns:
        glyph string representation
        
    Raises:
        ValueError: If charset is empty or contrast invalid
    """
    # Validation with specific feedback
    if not charset:
        raise ValueError("Character set cannot be empty")
    if not 0.1 <= contrast <= 3.0:
        raise ValueError(f"Contrast ({contrast}) must be between 0.1 and 3.0")
        
    # Apply contrast with proper scaling
    mid = 128
    adjusted = np.clip(((values - mid) * contrast + mid), 0, 255).astype(np.uint8)
    
    # Character mapping with direction control
    char_set = charset[::-1] if invert else charset
    
    # Efficient vectorized mapping - O(n) complexity
    char_indices = (adjusted * (len(char_set) - 1) / 255).astype(np.uint8)
    result = ''.join(char_set[idx] for idx in char_indices)
    
    return result
```

## 🧪 Testing Framework

Tests must be:

1. **Complete** - Covering functionality, edge cases, and error conditions
2. **Fast** - Executing without unnecessary delays
3. **Independent** - No test should depend on another
4. **Readable** - Clear test purpose and expectations
5. **Maintainable** - Easy to update with implementation changes

```python
# Example test structure
def test_map_luminance_to_glyph():
    # Setup
    values = np.array([0, 128, 255])
    charset = ".#@"
    
    # Core functionality
    result = map_luminance_to_glyph(values, charset)
    assert result == ".#@"
    
    # Edge cases
    assert map_luminance_to_glyph(np.array([]), charset) == ""
    
    # Input validation
    with pytest.raises(ValueError, match="Character set cannot be empty"):
        map_luminance_to_glyph(values, "")
```

## 📝 Documentation Standards

Documentation requirements:

1. **API Completeness** - Every public interface fully documented
2. **Example-Driven** - Practical usage examples for all capabilities
3. **Context-Aware** - Explaining not just how, but why
4. **Format Consistency** - Adhering to established patterns

## 📊 PR Review Criteria

Pull requests are evaluated on:

1. **Functional Correctness** - It must work as specified
2. **Test Coverage** - All code paths must be tested
3. **Documentation Quality** - Clear, complete, and accurate
4. **Performance Impact** - No unintended performance degradation
5. **Architectural Fit** - Alignment with system design

## 💡 Enhancement Focus Areas

Glyph Forge welcomes these improvements:

1. **Rendering Engine Optimizations** - Faster, more memory-efficient processing
2. **Character Mapping Systems** - New artistically-balanced character sets
3. **Format Support** - Additional input/output formats
4. **Terminal Compatibility** - Enhanced support across environments
5. **Integration Interfaces** - Simplified connections to other systems

## 🛠️ Development Tools

Required tools:

- **pytest** - Test execution
- **black** - Code formatting
- **isort** - Import organization
- **mypy** - Static type checking
- **flake8** - Style verification
- **coverage** - Test coverage analysis

## 🔗 Contacts & Recognition

- **Lloyd Handyside** (<ace1928@gmail.com>) — Implementation Lead
- **Eidos** (<syntheticeidos@gmail.com>) — Architectural Vision
- **Neuroforge** (<lloyd.handyside@neuroforge.io>) — Organization

Contributors recognized in CONTRIBUTORS.md and release notes.

---

"glyph art without structure is just random characters with hope."
    - ⚡ Glyph Forge Team ⚡
