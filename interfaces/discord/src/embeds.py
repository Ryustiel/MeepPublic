"""
Designs des dialogs de confirmation.
"""
import discord, json
from typing import Literal, Optional

def custom_embed(title: str, description: str, color: Optional[str] = None, image_url: Optional[str] = None) -> discord.Embed:
    """Crée un embed Discord personnalisé."""
    color = discord.Color.from_str(color) if color else discord.Color.default()
    embed = discord.Embed(title=title, description=description, color=color)
    if image_url:
        embed.set_footer(text=image_url)
    return embed

def cf_tool_call(tool_name: str, json_input: dict, allowed_users_string: str, security_icon_url: str) -> discord.Embed:
    """Affiche les arguments d'un tool call dans un embed Discord."""

    stringified_js = json.dumps(json_input, indent=4)  
    bounded_stringified_js = stringified_js[:min(4050, len(stringified_js))] # Limit to 4050 characters for Discord embed

    return (
        discord.Embed(
            title=tool_name.replace("_", " ").title(),
            description=f"```json\n{bounded_stringified_js}```\n",
            color=discord.Color.orange()
        )
        # .set_thumbnail(url="https://preview.redd.it/mn153n213viz.png?width=640&crop=smart&auto=webp&s=ab09fac88e5cafde7aa823d3ac398752d84e7c12")
        .set_footer(icon_url=security_icon_url, text=f"Action réservée à "+allowed_users_string)
    )

# TODO : Create embeds for inserting a root, a noun and a compound.

def cf_create_root(root: str, type: Literal["initial", "final"], description: str) -> discord.Embed:
    """Affiche les arguments d'une création de root dans un embed Discord."""
    
    return (
        discord.Embed(
            title=f"Ajouter `{root}`",
            description=f"Type: `{type}`\nDescription: `{description}`",
            color=discord.Color.green() if type == "initial" else discord.Color.purple()
        )
        .set_footer(icon_url="https://cdn.discordapp.com/avatars/361438727492337664/c92f6ec6a70d28896307064bfa8fbacb.png?size=1024", text=f"Action réservée à rouf")
    )
    
def cf_create_noun(noun: str, initial: str, final: str, description: str) -> discord.Embed:
    return (
        discord.Embed(
            title=f"Ajouter `{noun}`",
            description=f"Initial: `{initial}`\nFinal: `{final}`\nDescription: `{description}`",
            color=discord.Color.orange()
        )
        .set_footer(icon_url="https://cdn.discordapp.com/avatars/361438727492337664/c92f6ec6a70d28896307064bfa8fbacb.png?size=1024", text=f"Action réservée à rouf")
    )

def cf_create_compound(compound: str, translation: str) -> discord.Embed:
    return (
        discord.Embed(
            title=f"Nouveau compound",
            description=f"Phrase: `{compound}`\nInterprétation: `{translation}`",
            color=discord.Color.blue()
        )
        .set_footer(icon_url="https://cdn.discordapp.com/avatars/361438727492337664/c92f6ec6a70d28896307064bfa8fbacb.png?size=1024", text=f"Action réservée à rouf")
    )
