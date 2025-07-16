import React, { useState } from "react";
import { ChakraProvider, Flex, extendTheme } from "@chakra-ui/react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";

// Optional: custom font
const theme = extendTheme({
  fonts: {
    heading: `'Inter', 'Segoe UI', Arial, sans-serif`,
    body: `'Inter', 'Segoe UI', Arial, sans-serif`,
  },
});

const API_URL = "http://localhost:8000/chat";

export default function App() {
  const [chats, setChats] = useState([
    {
      id: 0,
      title: "New Chat",
      messages: [
        {
          sender: "bot",
          text: "👋 Welcome! You can ask me to buy/sell stocks, check prices, or view your portfolio.",
          ts: new Date(),
        },
      ],
    },
  ]);
  const [selectedChat, setSelectedChat] = useState(0);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (userText) => {
    if (!userText.trim()) return;
    const chatIndex = selectedChat;
    const userMsg = {
      sender: "user",
      text: userText,
      ts: new Date(),
    };
    const updatedChats = [...chats];
    updatedChats[chatIndex].messages.push(userMsg);
    setChats(updatedChats);
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });
      const data = await res.json();
      updatedChats[chatIndex].messages.push({
        sender: "bot",
        text: data.result || "No reply.",
        ts: new Date(),
      });
      setChats([...updatedChats]);
    } catch {
      updatedChats[chatIndex].messages.push({
        sender: "bot",
        text: "❌ Could not reach backend.",
        ts: new Date(),
      });
      setChats([...updatedChats]);
    }
    setLoading(false);
  };

  const addChat = () => {
    setChats([
      ...chats,
      {
        id: chats.length,
        title: "New Chat",
        messages: [
          {
            sender: "bot",
            text: "👋 New conversation started.",
            ts: new Date(),
          },
        ],
      },
    ]);
    setSelectedChat(chats.length);
  };

  return (
    <ChakraProvider theme={theme}>
      <Flex minH="100vh" height="100vh" bg="#000" fontFamily="'Inter', 'Segoe UI', Arial, sans-serif">
        <Sidebar
          chats={chats}
          selected={selectedChat}
          onSelect={setSelectedChat}
          onNewChat={addChat}
        />
        <Flex
          direction="column"
          flex="1"
          align="stretch" // no "center"
          justify="flex-start"
          bg="#fff"
          borderRadius="0"
          m={0}
          boxShadow="none"
          minH="100vh"
          height="100vh"
          width="100%"
        >
          <ChatWindow chat={chats[selectedChat]} loading={loading} />
          <ChatInput onSend={sendMessage} loading={loading} />
        </Flex>
      </Flex>
    </ChakraProvider>
  );
}
