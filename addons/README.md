# Chorus Engine Add-ons

This directory contains optional add-ons and setup scripts for Chorus Engine.

## Available Add-ons

### 🌟 Nova Setup (`nova-setup/`)

**Complete character configuration showcase**

Sets up the Nova character with all features enabled:
- Profile picture
- Voice sample for TTS cloning (Chatterbox)
- Image generation workflow (ComfyUI)
- Full character configuration

**Perfect for:**
- New users wanting to see all features in action
- Testing the complete Chorus Engine experience
- Learning how to configure characters

**Quick Start:**
```bash
# Windows
addons\nova-setup\setup_nova.bat

# Linux/Mac
./addons/nova-setup/setup_nova.sh
```

See `nova-setup/README.md` for detailed documentation.

---

## Creating Your Own Add-ons

Add-ons are a great way to:
- Share character configurations
- Distribute workflows and assets
- Provide setup automation
- Package themed collections

### Recommended Structure

```
addons/
  your-addon-name/
    README.md           # Documentation
    setup_script.py     # Automated setup (optional)
    setup.bat           # Windows helper (optional)
    setup.sh            # Linux/Mac helper (optional)
    assets/             # Images, audio, workflows, etc.
```

### Best Practices

1. **Include a README**: Explain what the add-on does and how to use it
2. **Provide automation**: Setup scripts make installation easier
3. **Bundle all assets**: Don't require external downloads
4. **Test thoroughly**: Verify on fresh installations
5. **Document prerequisites**: List any required services (ComfyUI, TTS, etc.)

### Asset Guidelines

- **Profile images**: 512x512 PNG or JPG
- **Voice samples**: MP3, 10-20 seconds, clear speech
- **Workflows**: Include ComfyUI JSON with `__CHORUS_PROMPT__` placeholder
- **Character YAMLs**: Include complete, tested configurations

---

## Contributing Add-ons

Want to share your character setup or workflow collection?

1. Create your add-on following the structure above
2. Test on a fresh Chorus Engine installation
3. Submit a pull request with:
   - Complete add-on folder
   - Clear README documentation
   - Any required attribution for assets

Good add-on ideas:
- Character packs (fantasy, sci-fi, historical, etc.)
- Workflow collections (art styles, specific use cases)
- Voice sample libraries
- Theme packages

---

## Support

For help with add-ons:
- Check the add-on's specific README
- Review main documentation in `Documentation/`
- Open an issue on GitHub

---

**Note**: Add-ons are optional and don't affect core Chorus Engine functionality. The engine works perfectly fine without any add-ons installed.
