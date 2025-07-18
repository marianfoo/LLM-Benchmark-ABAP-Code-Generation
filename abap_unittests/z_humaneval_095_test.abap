CLASS ltcl_test_check_dict_case DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_dict_entry,
             key   TYPE string,
             value TYPE string,
           END OF ty_dict_entry,
           tt_dict TYPE TABLE OF ty_dict_entry.
    METHODS: test_lowercase_keys FOR TESTING,
      test_mixed_case_keys FOR TESTING,
      test_numeric_key FOR TESTING,
      test_mixed_cases FOR TESTING,
      test_uppercase_keys FOR TESTING,
      test_capitalized_keys FOR TESTING,
      test_empty_dict FOR TESTING.

ENDCLASS.

CLASS ltcl_test_check_dict_case IMPLEMENTATION.

  METHOD test_lowercase_keys.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lt_dict = VALUE #( ( key = 'p' value = 'pineapple' )
                       ( key = 'b' value = 'banana' ) ).

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_true( act = lv_result ).
  ENDMETHOD.

  METHOD test_mixed_case_keys.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lt_dict = VALUE #( ( key = 'p' value = 'pineapple' )
                       ( key = 'A' value = 'banana' )
                       ( key = 'B' value = 'banana' ) ).

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_false( act = lv_result ).
  ENDMETHOD.

  METHOD test_numeric_key.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lt_dict = VALUE #( ( key = 'p' value = 'pineapple' )
                       ( key = '5' value = 'banana' )
                       ( key = 'a' value = 'apple' ) ).

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_false( act = lv_result ).
  ENDMETHOD.

  METHOD test_mixed_cases.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lt_dict = VALUE #( ( key = 'Name' value = 'John' )
                       ( key = 'Age' value = '36' )
                       ( key = 'City' value = 'Houston' ) ).

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_false( act = lv_result ).
  ENDMETHOD.

  METHOD test_uppercase_keys.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lt_dict = VALUE #( ( key = 'STATE' value = 'NC' )
                       ( key = 'ZIP' value = '12345' ) ).

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_true( act = lv_result ).
  ENDMETHOD.

  METHOD test_capitalized_keys.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lt_dict = VALUE #( ( key = 'fruit' value = 'Orange' )
                       ( key = 'taste' value = 'Sweet' ) ).

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_true( act = lv_result ).
  ENDMETHOD.

  METHOD test_empty_dict.
    DATA: lt_dict   TYPE tt_dict,
          lv_result TYPE abap_bool.

    lv_result = z_humaneval_095=>check_dict_case( lt_dict ).

    cl_abap_unit_assert=>assert_false( act = lv_result ).
  ENDMETHOD.

ENDCLASS.