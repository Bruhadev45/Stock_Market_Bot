import React from "react";
import { Box, VStack } from "@chakra-ui/react";

export default function ChatWindow({ chat, loading }) {
  return (
    <Box
      flex="1"
      width="100%"
      px={8} // No horizontal padding!
      py={6}
      bg="#fff"
      borderRadius="0"
      overflowY="auto"
      fontFamily="'Inter', 'Segoe UI', Arial, sans-serif"
      style={{ minHeight: "78vh" }}
    >
      <VStack spacing={3} align="stretch">
        {chat.messages.map((msg, i) => (
          <Box
            key={i}
            w="100%"
            display="flex"
            justifyContent={msg.sender === "user" ? "flex-end" : "flex-start"}
          >
            <Box
              maxW="70%"
              minW="25%"
              bg={msg.sender === "user" ? "#fff" : "#f5f5f5"}
              color="#111"
              px={5}
              py={3}
              borderRadius={msg.sender === "user" ? "2xl 2xl 2xl 0" : "2xl 2xl 0 2xl"}
              boxShadow="sm"
              fontWeight="500"
              fontSize="lg"
              border={msg.sender === "user" ? "1.5px solid #222" : "1.5px solid #eaeaea"}
              ml={msg.sender === "user" ? "auto" : 0}
              mr={msg.sender === "user" ? 0 : "auto"}
              textAlign="left"
            >
              {msg.text}
            </Box>
          </Box>
        ))}
        {loading && (
          <Box w="100%" display="flex" justifyContent="flex-start">
            <Box
              maxW="60%"
              bg="#f5f5f5"
              color="#222"
              px={5}
              py={3}
              borderRadius="2xl 2xl 0 2xl"
              fontWeight="500"
              border="1.5px solid #eaeaea"
              opacity={0.8}
              fontStyle="italic"
              ml={0}
              mr="auto"
            >
              Thinking...
            </Box>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
