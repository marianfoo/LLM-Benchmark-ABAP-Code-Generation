CLASS ltcl_test_digitsum DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_empty_string FOR TESTING,
      test_abab FOR TESTING,
      test_abccd FOR TESTING,
      test_helloe FOR TESTING,
      test_woarbld FOR TESTING,
      test_aaaaxxa FOR TESTING,
      test_how_are_you FOR TESTING,
      test_you_are_very_smart FOR TESTING.

ENDCLASS.

CLASS ltcl_test_digitsum IMPLEMENTATION.

  METHOD test_empty_string.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( '' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 0 ).
  ENDMETHOD.

  METHOD test_abab.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( 'abAB' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 131 ).
  ENDMETHOD.

  METHOD test_abccd.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( 'abcCd' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 67 ).
  ENDMETHOD.

  METHOD test_helloe.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( 'helloE' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 69 ).
  ENDMETHOD.

  METHOD test_woarbld.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( 'woArBld' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 131 ).
  ENDMETHOD.

  METHOD test_aaaaxxa.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( 'aAaaaXa' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 153 ).
  ENDMETHOD.

  METHOD test_how_are_you.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( ' How are yOu?' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 151 ).
  ENDMETHOD.

  METHOD test_you_are_very_smart.
    DATA: lv_result TYPE i.
    lv_result = z_humaneval_066=>digitsum( 'You arE Very Smart' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 327 ).
  ENDMETHOD.

ENDCLASS.