import React from "react";
import { Box, VStack, Button, Avatar, Text, Input, Divider } from "@chakra-ui/react";
import { FiPlus, FiSettings } from "react-icons/fi";

export default function Sidebar({ chats, selected, onSelect, onNewChat }) {
  return (
    <Box
      w={["70vw", "320px"]}
      bg="#000"
      color="#fff"
      h="100vh"
      p={4}
      roundedRight="2xl"
      shadow="lg"
      display="flex"
      flexDirection="column"
      justifyContent="space-between"
      fontFamily="'Inter', 'Segoe UI', Arial, sans-serif"
    >
      <Box>
        {/* App Title */}
        <Text fontSize="2xl" fontWeight="bold" mb={6} letterSpacing="1px" color="#fff">
          CHAT A.I+
        </Text>
        {/* New Chat Button */}
        <Button
          leftIcon={<FiPlus />}
          color="#fff"
          bg="#232323"
          _hover={{ bg: "#333" }}
          border="1px solid #444"
          w="100%"
          rounded="xl"
          mb={6}
          fontWeight="semibold"
          onClick={onNewChat}
          boxShadow="md"
        >
          New chat
        </Button>
        {/* Chat List */}
        <VStack align="stretch" spacing={1} maxH="60vh" overflowY="auto" mb={4}>
          {chats.map((chat, i) => (
            <Button
              key={chat.id}
              variant="ghost"
              bg={selected === i ? "#232323" : "transparent"}
              color="#fff"
              _hover={{ bg: "#181818" }}
              borderLeft={selected === i ? "4px solid #fff" : "4px solid transparent"}
              borderRadius="0"
              justifyContent="flex-start"
              px={4}
              py={3}
              fontWeight={selected === i ? "bold" : "normal"}
              fontSize="md"
              textAlign="left"
              onClick={() => onSelect(i)}
              boxShadow={selected === i ? "md" : "none"}
              transition="background 0.1s, font-weight 0.1s"
            >
              {chat.title}
            </Button>
          ))}
        </VStack>
      </Box>
      {/* Bottom User/Settings */}
    
        <Box display="flex" alignItems="center" mt={2} px={1}>
          <Avatar size="sm" name="You" bg="#fff" color="#000" mr={2} />
          <Text fontSize="sm" color="#eee">
            Bruhadev
          </Text>
        </Box>
      </Box>
  );
}
