CLASS z_humaneval_066 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: digitsum
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_066 IMPLEMENTATION.

  METHOD digitsum.
    DATA: lv_char  TYPE c LENGTH 1,
          lv_ascii TYPE i.

    IF s IS INITIAL.
      result = 0.
      RETURN.
    ENDIF.

    result = 0.

    DO strlen( s ) TIMES.
      DATA(lv_offset) = sy-index - 1.
      lv_char = s+lv_offset(1).

      IF lv_char CA 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.
        IF lv_char = 'A'. lv_ascii = 65. ENDIF.
        IF lv_char = 'B'. lv_ascii = 66. ENDIF.
        IF lv_char = 'C'. lv_ascii = 67. ENDIF.
        IF lv_char = 'D'. lv_ascii = 68. ENDIF.
        IF lv_char = 'E'. lv_ascii = 69. ENDIF.
        IF lv_char = 'F'. lv_ascii = 70. ENDIF.
        IF lv_char = 'G'. lv_ascii = 71. ENDIF.
        IF lv_char = 'H'. lv_ascii = 72. ENDIF.
        IF lv_char = 'I'. lv_ascii = 73. ENDIF.
        IF lv_char = 'J'. lv_ascii = 74. ENDIF.
        IF lv_char = 'K'. lv_ascii = 75. ENDIF.
        IF lv_char = 'L'. lv_ascii = 76. ENDIF.
        IF lv_char = 'M'. lv_ascii = 77. ENDIF.
        IF lv_char = 'N'. lv_ascii = 78. ENDIF.
        IF lv_char = 'O'. lv_ascii = 79. ENDIF.
        IF lv_char = 'P'. lv_ascii = 80. ENDIF.
        IF lv_char = 'Q'. lv_ascii = 81. ENDIF.
        IF lv_char = 'R'. lv_ascii = 82. ENDIF.
        IF lv_char = 'S'. lv_ascii = 83. ENDIF.
        IF lv_char = 'T'. lv_ascii = 84. ENDIF.
        IF lv_char = 'U'. lv_ascii = 85. ENDIF.
        IF lv_char = 'V'. lv_ascii = 86. ENDIF.
        IF lv_char = 'W'. lv_ascii = 87. ENDIF.
        IF lv_char = 'X'. lv_ascii = 88. ENDIF.
        IF lv_char = 'Y'. lv_ascii = 89. ENDIF.
        IF lv_char = 'Z'. lv_ascii = 90. ENDIF.
        result = result + lv_ascii.
      ENDIF.
    ENDDO.

  ENDMETHOD.

ENDCLASS.