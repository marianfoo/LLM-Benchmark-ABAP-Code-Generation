CLASS z_humaneval_017 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS parse_music
      IMPORTING
        VALUE(music_string) TYPE string
      RETURNING
        VALUE(result)       TYPE int4_table.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_017 IMPLEMENTATION.

  METHOD parse_music.
    DATA: lt_notes TYPE TABLE OF i,
          lv_note  TYPE i,
          lt_split TYPE TABLE OF string,
          lv_item  TYPE string.

    " Correct SPLIT syntax in ABAP
    SPLIT music_string AT ' ' INTO TABLE lt_split.

    LOOP AT lt_split INTO lv_item.
      CASE lv_item.
        WHEN 'o'.
          lv_note = 4.
        WHEN 'o|'.
          lv_note = 2.
        WHEN '.|'.
          lv_note = 1.
        WHEN OTHERS.
          CONTINUE.
      ENDCASE.

      APPEND lv_note TO lt_notes.
    ENDLOOP.

    result = lt_notes.
  ENDMETHOD.

ENDCLASS.