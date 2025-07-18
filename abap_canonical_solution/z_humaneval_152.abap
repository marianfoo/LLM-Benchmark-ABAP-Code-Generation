CLASS z_humaneval_152 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS compare
      IMPORTING
        game          TYPE int4_table
        guess         TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_152 IMPLEMENTATION.

  METHOD compare.
    LOOP AT game INTO DATA(game_value).
      DATA(index) = sy-tabix.
      READ TABLE guess INTO DATA(guess_value) INDEX index.
      result = VALUE #( BASE result ( abs( game_value - guess_value ) ) ).
    ENDLOOP.
  ENDMETHOD.

ENDCLASS.