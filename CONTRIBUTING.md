# Contributing to ML Model Tensor Debugger

Welcome to the ML Model Tensor Debugger! This project follows **AI-native development practices**, meaning most features and bug fixes are implemented through AI-assisted development.

## 🤖 AI-Native Development

This repository is designed for **AI-native contribution workflows**. Here's what that means:

### **What is AI-Native Development?**
- Features are developed through conversational programming with AI assistants (like Claude)
- Code changes are iterative and guided by natural language descriptions
- Documentation and implementation happen simultaneously  
- Testing and debugging are done through AI-assisted problem solving

### **Why AI-Native?**
- **Faster iteration**: Natural language requirements → working code
- **Better documentation**: AI naturally explains what it's building
- **Consistent patterns**: AI maintains architectural consistency
- **Accessible contribution**: Lower barrier to entry for complex features

## 🛠️ Contribution Guidelines

### **For New Features**

When adding new features to this repository, please follow this process:

#### 1. **Document Your AI Session**
Create a markdown file in `ai_docs/sessions/` with the format: `YYYY-MM-DD-feature-name.md`

**Required sections:**
```markdown
# Feature: [Feature Name]
Date: YYYY-MM-DD
AI Assistant: [Claude/GPT-4/etc.]

## Original Request
[Copy the exact prompt/request that started the feature development]

## Key Implementation Details
[Major technical decisions made during development]

## Files Modified
- `file1.py` - [what was changed]
- `file2.html` - [what was changed]

## Testing Performed  
[What testing was done to verify the feature works]

## Known Issues
[Any limitations or issues discovered during development]
```

#### 2. **Prompt Engineering Documentation**
For complex features, include key prompts and AI responses:

```markdown
## Key AI Interactions

### Prompt: Multi-dimensional slider implementation
**Human:** "Update the sliders to take the model shape into account. If there are three dimensions, the slider should apply to any dimension."

**AI Response:** [Include the AI's analysis and approach]

**Result:** [What code was generated and how it solved the problem]
```

### **For Bug Fixes**

#### 1. **Reference the Bug Report**
- Link to the issue in `KNOWN_BUGS.md` or GitHub Issues
- Include the bug reproduction case in your session documentation

#### 2. **Document the Debugging Process**
```markdown
## Debugging Session

### Problem Description
[Clear description of the bug]

### AI-Assisted Diagnosis  
**Prompt:** "There is an error: [error message]. The shapes are [shape details]."
**AI Analysis:** [How the AI identified the root cause]

### Solution Implementation
[Step-by-step fix with AI guidance]

### Verification
[How the fix was tested and verified]
```

### **Example AI Session Documentation**

See `ai_docs/sessions/2025-08-06-multidimensional-sliders.md` for a complete example of AI-native feature development documentation.

## 📋 Development Workflow

### **Recommended Process**
1. **Start with natural language**: Describe what you want to build
2. **Iterate with AI**: Use conversational development to refine the implementation
3. **Document as you go**: Capture key decisions and AI insights
4. **Test incrementally**: Verify each step works before moving on
5. **Update documentation**: Ensure README and other docs reflect changes

### **File Organization**
```
model-tensor-debugger/
├── ai_docs/
│   ├── HOW_IT_WORKS.md          # Architecture documentation
│   └── sessions/                # AI development session logs
│       ├── 2025-08-06-multidimensional-sliders.md
│       ├── 2025-08-07-cpu-only-tensors.md
│       └── [date]-[feature].md
├── app.py                       # Main Flask backend
├── templates/index.html         # Frontend interface
├── KNOWN_BUGS.md               # Bug tracking
├── CONTRIBUTING.md             # This file
└── README.md                   # User documentation
```

## 🎯 Contribution Areas

### **High-Impact Contributions**
- **Bug fixes**: Address issues in `KNOWN_BUGS.md`
- **Performance optimization**: Improve tensor loading/processing speed
- **UI/UX improvements**: Better visualization and navigation
- **TP compatibility**: Support for more tensor parallel configurations

### **Documentation Contributions**
- **Session documentation**: Add your AI development sessions
- **Architecture updates**: Improve `HOW_IT_WORKS.md`
- **User guides**: Enhance README with more examples
- **Bug documentation**: Help track and categorize issues

### **Testing Contributions**  
- **Edge case testing**: Find and document new tensor shape combinations
- **Performance testing**: Test with larger models and more complex scenarios
- **User testing**: Gather feedback from real debugging sessions

## 🧪 Testing Guidelines

Since this is an AI-native project, testing often happens through iterative conversation:

### **Manual Testing Process**
1. **Describe the test case** to your AI assistant
2. **Generate test data** or use existing tensor files  
3. **Document expected vs actual behavior**
4. **Iterate on fixes** through AI-assisted debugging

### **Areas to Test**
- Different tensor shapes and TP configurations
- Large tensor files (memory usage)
- Edge cases (empty tensors, NaN values, etc.)
- Browser compatibility (Chrome, Firefox, Safari)
- Mobile responsiveness

## 💡 AI-Assisted Development Tips

### **Effective Prompting**
- **Be specific**: Include exact error messages and tensor shapes
- **Provide context**: Reference existing code structure and patterns
- **Ask for explanations**: Understanding the "why" helps with future changes
- **Request documentation**: Ask AI to explain complex implementations

### **Good AI Development Practices**
- **Incremental changes**: Make small, testable changes
- **Maintain consistency**: Follow existing code patterns and naming
- **Document decisions**: Capture why certain approaches were chosen  
- **Test edge cases**: Ask AI to identify potential failure scenarios

### **Example Prompts**
```
Good: "The tensor slicing fails when shapes are [1, 8, 76, 64] vs [76, 8, 64]. 
The error shows 'slice failed, falling back to flattened view'. How should we 
fix the dimension mapping in the slider creation?"

Better: "Looking at app.py line 720, the dimension slicing creates indices 
that don't match the reshaped tensor. The original shapes get reshaped by 
smart_reshape_for_tp(), but the frontend still uses original dimensions. 
What's the best architectural approach to fix this mismatch?"
```

## 🔄 Review Process

### **Self-Review Checklist**
- [ ] AI session documented in `ai_docs/sessions/`
- [ ] Key prompts and responses captured
- [ ] Files modified and changes described  
- [ ] Testing performed and results documented
- [ ] Known limitations identified
- [ ] Existing documentation updated

### **Code Quality Standards**
- **Consistency**: Follow existing patterns in the codebase
- **Documentation**: Comprehensive inline comments for complex logic
- **Error handling**: Robust error handling with user-friendly messages
- **Performance**: Consider memory usage and processing time

## 🤝 Community Guidelines

### **AI-Native Collaboration**
- **Share prompts**: Good prompts benefit everyone  
- **Document discoveries**: New insights should be captured
- **Iterate openly**: Show your development process, including false starts
- **Learn from others**: Review other contributors' AI sessions

### **Communication Style**
- **Technical precision**: Include exact error messages and reproduction steps
- **Process transparency**: Show how you worked with AI to solve problems
- **Knowledge sharing**: Explain complex solutions for future contributors

## 📚 Resources

### **AI Development Resources**
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Effective AI Prompting Guide](https://www.anthropic.com/prompting)
- PyTorch Documentation for tensor operations

### **Project-Specific Resources**
- [`ai_docs/HOW_IT_WORKS.md`](ai_docs/HOW_IT_WORKS.md) - Architecture overview
- [`KNOWN_BUGS.md`](KNOWN_BUGS.md) - Current issues and planned fixes
- [`README.md`](README.md) - User documentation and features

---

## 🎉 Get Started

Ready to contribute? Here's how to begin:

1. **Set up your development environment** (see README.md)
2. **Pick an issue** from `KNOWN_BUGS.md` or propose a new feature
3. **Start your AI session** with a clear problem description
4. **Document your process** as you develop
5. **Test your changes** and capture the results
6. **Submit your contribution** with complete AI session documentation

Welcome to AI-native development! 🚀