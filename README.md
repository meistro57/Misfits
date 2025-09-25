# 🏠✨ Misfits! ✨🏠

<div align="center">

**"No scripts. Just chaos."**

*An AI-powered life simulation where quirky characters create their own stories*

[![Open Source](https://img.shields.io/badge/Open%20Source-💚-brightgreen)](https://github.com/your-org/misfits)
[![AI Powered](https://img.shields.io/badge/AI%20Powered-🤖-blue)](https://github.com/your-org/misfits)
[![Community Driven](https://img.shields.io/badge/Community%20Driven-👥-purple)](https://github.com/your-org/misfits)
[![Built with Love](https://img.shields.io/badge/Built%20with-❤️-red)](https://github.com/your-org/misfits)

</div>

---

## 🌟 What Makes Misfits! Special?

Forget everything you know about life simulation games. **Misfits!** isn't about managing needs bars or following scripted storylines. It's about stepping back and watching genuinely intelligent AI characters live their own chaotic, beautiful, messy lives.

### 🎭 **Meet the Misfits**
Each character is powered by their own AI brain with:
- **Evolving personalities** that grow from experiences
- **Persistent memories** of every interaction, grudge, and friendship
- **Hidden desires** that create internal conflicts and drama
- **Unscripted dialogue** generated fresh every time they speak

### 🎪 **Pure Emergent Chaos**
- Watch a paranoid-but-charismatic neighbor accidentally start a cult
- Witness a romantic nihilist fall in love despite believing nothing matters
- See friendships bloom and crumble based on who ate whose sandwich
- Experience completely unique soap operas that write themselves

### 🎮 **You're the Architect of Chaos**
Instead of micromanaging characters, you:
- **Build the stage** with houses, objects, and environments
- **Whisper suggestions** directly into characters' minds
- **Trigger random events** with the glorious Chaos Button
- **Watch stories unfold** that surprise even you

---

## 🚀 Quick Start

```bash
# Clone the chaos
git clone https://github.com/your-org/misfits-game.git
cd misfits-game

# Set up your world
pip install -r requirements.txt
python scripts/setup.py

# Let the chaos begin!
python main.py
```

**First time?** Check out our [🎯 Quick Start Guide](docs/quick-start.md) to create your first neighborhood of digital misfits!

---

## 🎨 Game Modes: Pick Your Flavor of Chaos

<table>
<tr>
<td width="50%">

### 🤡 **Comedy & Chaos Mode**
*Maximum absurdity and slapstick drama*

Perfect for when you want to laugh until your sides hurt. Characters pull pranks, start ridiculous feuds, and turn mundane situations into comedy gold.

**Chaos Level:** 🔥🔥🔥🔥🔥

</td>
<td width="50%">

### 🧠 **Psychological & Deep Mode**  
*Realistic emotions and complex relationships*

Watch characters develop trust issues, work through trauma, and form deep, meaningful connections. Every action has long-term psychological consequences.

**Depth Level:** 🌊🌊🌊🌊🌊

</td>
</tr>
<tr>
<td width="50%">

### 📚 **Learning & Growth Mode**
*Characters that adapt and evolve*

A living laboratory where AI characters learn new skills, develop habits, and grow from their experiences. Perfect for AI enthusiasts and researchers.

**Evolution Level:** 🔬🔬🔬🔬🔬

</td>
<td width="50%">

### ⚙️ **Sandbox Mode**
*Complete creative control*

Mix and match chaos levels, realism settings, and drama intensities. Create custom worlds that blend comedy with psychology, or pure chaos with deep learning.

**Control Level:** 🎛️🎛️🎛️🎛️🎛️

</td>
</tr>
</table>

---

## 🧠 The Magic Behind the Mayhem

### **AI Personality Engine** 🤖
```
Every Misfit = Unique AI Brain + Complex Personality + Personal History
```
- **No scripted responses** - every conversation is generated fresh
- **Trait mashups** create unexpected personalities (Lazy + Overachiever = Eternal Burnout)
- **Hidden motivations** drive characters toward goals they might not even admit

### **Persistent Memory System** 🧬
```
Every interaction → Stored forever → Influences future decisions
```
- Characters remember **everything**: favors, betrayals, inside jokes, embarrassing moments
- **Gossip networks** spread rumors (true and false) throughout the neighborhood  
- **Legacy saves** let character memories haunt future playthroughs

### **Intervention Powers** ⚡
```
Subtle influence > Direct control = More interesting stories
```
- **Whisper suggestions** into characters' minds without forcing actions
- **Environmental changes** create new opportunities for drama
- **The Chaos Button** triggers delightfully unpredictable events

---

## 🎪 What Players Are Saying

> *"I watched my shy bookworm Misfit accidentally become the neighborhood's relationship counselor after overhearing one argument. Three hours later, there were marriage proposals and a love triangle I never saw coming."* 
> 
> — **Sarah M., Beta Tester**

> *"The AI is scary good. My 'reformed party animal' character kept making subtle references to their wild past, and when I checked their memory logs, there was this whole elaborate backstory they'd generated about college shenanigans."*
> 
> — **Mike R., AI Enthusiast**

> *"I thought I was just placing a hot tub. Six months of gameplay later, it had become the center of a neighborhood political movement about 'communal relaxation rights.' The characters are insane and I love them."*
> 
> — **Jessica L., Longtime Player**

---

## 🛠️ Built for Creators

### **100% Open Source** 📖
- **MIT License** - Use it, modify it, sell it, whatever
- **No paywalls** - Every feature is free forever
- **Community-driven** - The best ideas come from players like you

### **Modding Made Easy** 🔧
```json
// Create a new personality in minutes
{
  "conspiracy_theorist": {
    "traits": ["paranoid", "charismatic", "brilliant"],
    "hidden_desires": ["validation", "control"],
    "quirks": ["connects_everything", "collects_evidence"]
  }
}
```

### **Extensible Everything** 🚀
- **Custom personalities** via simple JSON files
- **New chaos events** with drag-and-drop simplicity  
- **Interactive objects** that trigger unique AI behaviors
- **Total conversion mods** supported and encouraged

---

## 🏗️ Technical Architecture

<details>
<summary><strong>🔍 Click to see the tech stack powering the chaos</strong></summary>

### **Core Technologies**
| Component | Technology | Why We Chose It |
|-----------|------------|-----------------|
| **Game Engine** | Godot 4.x | Open-source, lightweight, perfect for indie projects |
| **AI Brains** | Local LLMs (Ollama/LM Studio) | Privacy-focused, customizable, no API costs |
| **Memory** | SQLite + FAISS | Fast vector similarity search for AI memories |
| **Audio** | TTS + Lip-sync | Dynamic voices that match AI-generated personalities |
| **Config** | JSON/YAML | Human-readable, easy to mod |

### **System Architecture**
```
Player Input → World Tick → AI Processing → Memory Update → Visual Output
     ↑                                                            ↓
Environment ← ← ← Character Actions ← ← ← LLM Decisions ← ← ← Context
```

### **AI Decision Process**
1. **Gather Context**: Current needs + Relevant memories + Social relationships
2. **Generate Response**: Local LLM processes context into character action
3. **Execute Action**: Game world updates based on AI decision
4. **Store Memory**: Experience becomes part of character's permanent history

</details>

---

## 🚦 Getting Started

### **System Requirements**
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 8GB (16GB recommended for larger neighborhoods)
- **Storage**: 2GB free space
- **GPU**: Not required, but helps with larger AI models

### **Installation Options**

<table>
<tr>
<td width="33%">

#### 🚀 **Quick Start**
```bash
git clone [repo]
pip install -r requirements.txt
python main.py
```
*Best for developers*

</td>
<td width="33%">

#### 📦 **Pre-built Release**
Download from [Releases](releases)
- Windows .exe
- macOS .dmg  
- Linux AppImage
*Best for players*

</td>
<td width="33%">

#### 🐳 **Docker**
```bash
docker run -it misfits-game
```
*Best for servers*

</td>
</tr>
</table>

### **Your First 5 Minutes**
1. **Launch** the game and select **Comedy & Chaos Mode**
2. **Create** 3-4 Misfits with clashing personalities
3. **Build** a simple house with shared spaces
4. **Hit the Chaos Button** and grab some popcorn 🍿
5. **Watch** your digital soap opera begin!

---

## 🤝 Join the Community

### **Get Involved**
- 💬 **Discord**: Chat with other players and developers
- 🐛 **Issues**: Found a bug or have an idea? Let us know!
- 🔧 **Pull Requests**: Code contributions always welcome
- 📖 **Wiki**: Community-maintained guides and tips
- 🎨 **Mod Sharing**: Share your custom personalities and events

### **Contributing**
```bash
# Fork the repo
# Create a feature branch
git checkout -b amazing-new-feature

# Make your magic happen
# (Don't forget tests!)

# Submit a pull request
# Watch the chaos spread!
```

### **Community Guidelines**
- **Be kind** - We're all here to have fun with AI chaos
- **Share knowledge** - Help others create better Misfits
- **Test thoroughly** - Broken AI can create... interesting... problems
- **Document everything** - Future you will thank present you

---

## 🗺️ Roadmap

### **Version 1.0 - "The Great Awakening"** 🎯 *Current Focus*
- ✅ Core AI personality engine
- ✅ Persistent memory system  
- 🚧 Four simulation modes
- 🚧 Basic modding framework
- 🚧 Community tools

### **Version 1.1 - "Community Chaos"** 🎪 *Next Up*
- 📋 Enhanced mod creation tools
- 📋 Personality marketplace
- 📋 Advanced chaos events
- 📋 Multi-language support
- 📋 Performance optimizations

### **Version 2.0 - "Expansion Universe"** 🌌 *Future Vision*
- 📋 **Misfits: Apocalypse** - Survival mode expansion
- 📋 **Misfits: Utopia** - Harmony mode (until chaos breaks it)
- 📋 **Misfits: Cybernetics** - Augmented AI personalities
- 📋 Multiplayer neighborhoods
- 📋 VR support

---

## 📜 License & Credits

### **Open Source Forever** 
**MIT License** - See [LICENSE](LICENSE) for full details

This means you can:
- ✅ Use it commercially
- ✅ Modify the source code  
- ✅ Distribute your changes
- ✅ Place warranty (if you want)

### **Built With Love By**
- **Mark** - Creator of Eli GPT and Awakening Mind GPT
- **The Open Source Community** - Contributors, testers, and chaos enthusiasts
- **You** - For being part of this wild experiment!

### **Special Thanks**
- The Godot Foundation for an amazing engine
- The local LLM community for making AI accessible
- Every beta tester who survived the early chaos
- Coffee, for making late-night coding sessions possible ☕

---

## 🔮 Why Misfits! Matters

In a world of formulaic games and predictable AI, **Misfits!** represents something different:

**🧠 Genuine AI Creativity** - Watch characters surprise you with decisions no programmer scripted

**🎭 Emergent Storytelling** - Every playthrough creates unique narratives that couldn't exist anywhere else  

**🌍 Open Source Innovation** - A community-driven project that belongs to everyone

**🎪 Pure, Unfiltered Fun** - Sometimes the best games are about watching chaos unfold

---

<div align="center">

## 🚀 Ready to Meet Your Misfits?

**[📥 Download Now](releases)** • **[📖 Documentation](docs/)** • **[💬 Join Discord](discord-link)** • **[🐛 Report Issues](issues)**

*Built with ❤️ and a healthy dose of chaos*

---

**⭐ Like what you see? Give us a star and help spread the chaos!**

*Remember: In Misfits!, the best stories are the ones no human ever wrote.*

</div>
