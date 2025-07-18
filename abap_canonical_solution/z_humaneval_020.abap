CLASS z_humaneval_020 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_number,
             value TYPE f,
           END OF ty_number.
    TYPES: tt_numbers TYPE STANDARD TABLE OF ty_number.
    TYPES: tt_result TYPE SORTED TABLE OF f WITH UNIQUE KEY table_line.

    CLASS-METHODS: find_closest_elements
      IMPORTING
        VALUE(numbers)      TYPE tt_numbers
      RETURNING
        VALUE(closest_pair) TYPE tt_result.

ENDCLASS.

CLASS z_humaneval_020 IMPLEMENTATION.

  METHOD find_closest_elements.

    DATA: lv_min_distance TYPE f VALUE '999999',
          lv_distance     TYPE f,
          lv_first        TYPE f,
          lv_second       TYPE f,
          wa_i            TYPE ty_number,
          wa_j            TYPE ty_number,
          lv_idx          TYPE i,
          lv_lines        TYPE i.

    CLEAR closest_pair.

    lv_lines = lines( numbers ).

    CHECK lv_lines >= 2.

    LOOP AT numbers INTO wa_i.
      lv_idx = sy-tabix.
      LOOP AT numbers INTO wa_j FROM ( lv_idx + 1 ) TO lv_lines.
        lv_distance = abs( wa_i-value - wa_j-value ).
        IF lv_distance < lv_min_distance.
          lv_min_distance = lv_distance.
          lv_first = wa_i-value.
          lv_second = wa_j-value.
        ENDIF.
      ENDLOOP.
    ENDLOOP.

    IF lv_first < lv_second.
      INSERT lv_first INTO TABLE closest_pair.
      INSERT lv_second INTO TABLE closest_pair.
    ELSE.
      INSERT lv_second INTO TABLE closest_pair.
      INSERT lv_first INTO TABLE closest_pair.
    ENDIF.

  ENDMETHOD.

ENDCLASS.